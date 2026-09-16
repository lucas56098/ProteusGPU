// implements the cooling (cooling.h)

#include "../global/allvars.h"
#include "../io/input.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "astro_constants.h"
#include "cooling.h"
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace astro {

#ifdef COOLING

    HD double   tef_Y(double T, const CoolingTable& t);
    HD double   tef_Yinv(double Yt, const CoolingTable& t);
    HD void     cool_cell(uint64_t i, hydro::primvars* primvar, CoolingTable t, double dt_half);
    static void load_table(const std::string& path, CoolingTable& t);

    static CoolingTable g_cool;

    // reads the table and builds the unit factors
    void cooling_init() {
        load_table(input.get_parameter("cooling_table"), g_cool);
        g_cool.T_floor = input.get_parameter_double("T_floor");

        g_cool.C_T = (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * PROTONMASS * units.UnitVelocity_in_cm_per_s *
                     units.UnitVelocity_in_cm_per_s / BOLTZMANN;

        g_cool.C_dY = (g_cool.L_ref / g_cool.T_ref) * (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * HYDROGEN_MASSFRAC *
                      units.UnitDensity_in_cgs() * units.UnitTime_in_s() / (MEAN_MOL_WEIGHT_E * BOLTZMANN * PROTONMASS);

        sim.min_egy_spec = g_cool.T_floor / g_cool.C_T;

        logging::root() << "COOLING: loaded " << g_cool.N << " node table, T = [" << g_cool.T[0] << ", " << g_cool.T_ref
                        << "] K, floor " << g_cool.T_floor << " K" << std::endl;
    }

    void cooling_apply(double dt_half) {
        PROFILE("COOLING");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;

        const CoolingTable t = g_cool;

        parallel_for<_HYDRO_BLOCK_SIZE_>(
            "COOL_KERNEL", mesh->n_hydro, [=] HD(size_t i) { cool_cell(i, primvar, t, dt_half); });
    }

    // cools one cell over dt_half, in one step and without a substep loop
    HD void cool_cell(uint64_t i, hydro::primvars* primvar, CoolingTable t, double dt_half) {
        const double rho = primvar->rho[i];
        POINT_TYPE   v   = primvar->v[i];

        double v2 = v.x * v.x + v.y * v.y;
#ifdef dim_3D
        v2 += v.z * v.z;
#endif
        const double e_int = primvar->E[i] - 0.5 * rho * v2;
        if (e_int <= 0.0) return;

        // temperature of the cell
        const double T = t.C_T * (e_int / rho);
        if (T <= t.T_floor) return;

        // cooling is exact in Y: walk a step in Y, then go back to a temperature
        double T_new = tef_Yinv(tef_Y(T, t) + t.C_dY * rho * dt_half, t);
        if (T_new < t.T_floor) T_new = t.T_floor;

        primvar->E[i] += rho * (T_new - T) / t.C_T;
    }

    // row below T
    HD static int segment_for_T(double T, const CoolingTable& t) {
        for (int k = 0; k < t.N - 1; k++) {
            if (T < t.T[k + 1]) return k;
        }
        return t.N - 2;
    }

    // Y of a temperature: the cooling time from T down, in units of the reference row
    HD double tef_Y(double T, const CoolingTable& t) {
        const int    k     = segment_for_T(T, t);
        const double a     = t.alpha[k];
        const double ratio = T / t.T[k];
        const double pref  = (t.L_ref / t.T_ref) * t.T[k] / t.L[k];
        // a slope of exactly 1 would divide by zero, there the integral is a logarithm
        const double term = (fabs(1.0 - a) > 1e-6) ? pref / (1.0 - a) * (pow(ratio, 1.0 - a) - 1.0) : pref * log(ratio);
        return t.Y[k] - term;
    }

    // and back: the temperature that belongs to a Y
    HD double tef_Yinv(double Yt, const CoolingTable& t) {
        if (Yt >= t.Y[0]) return t.T[0];

        int k = t.N - 2;
        for (int j = 0; j < t.N - 1; j++) {
            if (Yt >= t.Y[j + 1]) {
                k = j;
                break;
            }
        }
        const double a   = t.alpha[k];
        const double c   = (t.Y[k] - Yt) * (t.T_ref * t.L[k]) / (t.L_ref * t.T[k]);
        const double rat = (fabs(1.0 - a) > 1e-6) ? pow(1.0 + (1.0 - a) * c, 1.0 / (1.0 - a)) : exp(c);
        return t.T[k] * rat;
    }

    // reads rows of temperature and cooling rate, ignoring comments and empty lines
    static void load_table(const std::string& path, CoolingTable& t) {
        std::ifstream f(path);
        if (!f) { proteus_mpi::exit_failure("COOLING: cannot open cooling_table '%s'\n", path.c_str()); }

        std::vector<double> Tv, Lv;
        std::string         line;
        while (std::getline(f, line)) {
            const size_t s = line.find_first_not_of(" \t\r\n");
            if (s == std::string::npos || line[s] == '#') continue;
            std::istringstream iss(line);
            double             Ti, Li;
            if (iss >> Ti >> Li) {
                Tv.push_back(Ti);
                Lv.push_back(Li);
            }
        }
        if (Tv.size() < 2) { proteus_mpi::exit_failure("COOLING: table '%s' needs >= 2 rows\n", path.c_str()); }

        t.N     = (int)Tv.size();
        t.T     = gpu_alloc<double>(t.N);
        t.L     = gpu_alloc<double>(t.N);
        t.alpha = gpu_alloc<double>(t.N - 1);
        t.Y     = gpu_alloc<double>(t.N);
        for (int k = 0; k < t.N; k++) {
            t.T[k] = Tv[k];
            t.L[k] = Lv[k];
        }
        t.T_ref = t.T[t.N - 1];
        t.L_ref = t.L[t.N - 1];

        // slope of each segment on a log scale
        for (int k = 0; k < t.N - 1; k++) {
            t.alpha[k] = log(t.L[k + 1] / t.L[k]) / log(t.T[k + 1] / t.T[k]);
        }

        // Y of every row, summed up from the hottest one down
        t.Y[t.N - 1] = 0.0;
        for (int k = t.N - 2; k >= 0; k--) {
            const double a     = t.alpha[k];
            const double ratio = t.T[k + 1] / t.T[k];
            const double pref  = (t.L_ref / t.T_ref) * t.T[k] / t.L[k];
            const double seg =
                (fabs(1.0 - a) > 1e-6) ? pref / (1.0 - a) * (pow(ratio, 1.0 - a) - 1.0) : pref * log(ratio);
            t.Y[k] = t.Y[k + 1] + seg;
        }
    }

#endif

} // namespace astro
