/* optically-thin radiative cooling via Townsend (2009) exact integration */
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

    // forward declarations
    HD double tef_Y(double T, const CoolingTable& t);
    HD double tef_Yinv(double Yt, const CoolingTable& t);
    HD void   cool_cell(hsize_t i, hydro::primvars* primvar, CoolingTable t, double dt_half);
#ifndef CPU_DEBUG
    GLOBAL void kernel_cool(hsize_t n_hydro, hydro::primvars* primvar, CoolingTable t, double dt_half);
#endif
    static void load_table(const std::string& path, CoolingTable& t);

    static CoolingTable g_cool;

    // ============================================================
    // Setup
    // ============================================================

    void cooling_init() {
        load_table(input.getParameter("cooling_table"), g_cool);
        g_cool.T_floor = input.getParameterDouble("T_floor");

        // T[K] = C_T * (e_int_code / rho_code)
        g_cool.C_T = (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * PROTONMASS *
                     units.UnitVelocity_in_cm_per_s * units.UnitVelocity_in_cm_per_s / BOLTZMANN;

        // TEF step dY = C_dY * rho_code * dt_code, from dY = (L_ref/T_ref)*(n_e n_H / a)*dt with
        // a = rho k_B / ((gamma-1) mu m_H) and n_e n_H = (X_H/mu_e)(rho/m_H)^2
        g_cool.C_dY = (g_cool.L_ref / g_cool.T_ref) * (gamma_eos - 1.0) * MEAN_MOL_WEIGHT *
                      HYDROGEN_MASSFRAC * units.UnitDensity_in_cgs() * units.UnitTime_in_s() /
                      (MEAN_MOL_WEIGHT_E * BOLTZMANN * PROTONMASS);

        // let the hydro update enforce the same temperature floor (e_int >= rho * T_floor / C_T)
        sim.min_egy_spec = g_cool.T_floor / g_cool.C_T;

        logging::root() << "COOLING: loaded " << g_cool.N << " node table, T = [" << g_cool.T[0] << ", "
                        << g_cool.T_ref << "] K, floor " << g_cool.T_floor << " K" << std::endl;
    }

    // ============================================================
    // Apply
    // ============================================================

    void cooling_apply(double dt_half) {
        PROFILE("COOLING");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;

#ifndef CPU_DEBUG
        const int tpb    = _HYDRO_BLOCK_SIZE_;
        const int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("COOL_KERNEL");
            kernel_cool<<<blocks, tpb>>>(mesh->n_hydro, primvar, g_cool, dt_half);
            GPU_SYNC();
        }
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            cool_cell(i, primvar, g_cool, dt_half);
        }
#endif
    }

#ifndef CPU_DEBUG
    GLOBAL void kernel_cool(hsize_t n_hydro, hydro::primvars* primvar, CoolingTable t, double dt_half) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        cool_cell(i, primvar, t, dt_half);
    }
#endif

    // ============================================================
    // Per-cell work
    // ============================================================

    // Townsend update of one cell's internal energy; rho and velocity are untouched.
    HD void cool_cell(hsize_t i, hydro::primvars* primvar, CoolingTable t, double dt_half) {
        const double rho = primvar->rho[i];
        POINT_TYPE   v   = primvar->v[i];

        double v2 = v.x * v.x + v.y * v.y;
#ifdef dim_3D
        v2 += v.z * v.z;
#endif
        const double e_int = primvar->E[i] - 0.5 * rho * v2;
        if (e_int <= 0.0) return;

        const double T = t.C_T * (e_int / rho);
        if (T <= t.T_floor) return; // already at/below the floor: cooling is off

        double T_new = tef_Yinv(tef_Y(T, t) + t.C_dY * rho * dt_half, t);
        if (T_new < t.T_floor) T_new = t.T_floor;

        primvar->E[i] += rho * (T_new - T) / t.C_T; // change in e_int (T_new <= T -> energy removed)
    }

    // ============================================================
    // Townsend temporal evolution function Y and its inverse
    // ============================================================

    // segment holding temperature T (clamped to the top segment when T is above the table)
    HD static int segment_for_T(double T, const CoolingTable& t) {
        for (int k = 0; k < t.N - 1; k++) {
            if (T < t.T[k + 1]) return k;
        }
        return t.N - 2;
    }

    // Y(T) = Y_k - (L_ref/T_ref) * (T_k/(L_k(1-a))) * ((T/T_k)^(1-a) - 1)  [ln form when a==1 ]
    HD double tef_Y(double T, const CoolingTable& t) {
        const int    k     = segment_for_T(T, t);
        const double a     = t.alpha[k];
        const double ratio = T / t.T[k];
        const double pref  = (t.L_ref / t.T_ref) * t.T[k] / t.L[k];
        const double term  = (fabs(1.0 - a) > 1e-6) ? pref / (1.0 - a) * (pow(ratio, 1.0 - a) - 1.0)
                                                    : pref * log(ratio);
        return t.Y[k] - term;
    }

    // invert Y: find the segment bracketing Yt (Y decreases with node index), then solve for T
    HD double tef_Yinv(double Yt, const CoolingTable& t) {
        if (Yt >= t.Y[0]) return t.T[0]; // colder than the floor node

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

    // ============================================================
    // Table loading
    // ============================================================

    // read a two-column "T[K] Lambda[erg cm^3/s]" table (ascending T, '#' comments), then
    // precompute the per-segment slopes and the TEF nodes.
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

        // per-segment power-law slopes
        for (int k = 0; k < t.N - 1; k++) {
            t.alpha[k] = log(t.L[k + 1] / t.L[k]) / log(t.T[k + 1] / t.T[k]);
        }

        // TEF nodes, accumulated downward from Y(T_ref) = 0
        t.Y[t.N - 1] = 0.0;
        for (int k = t.N - 2; k >= 0; k--) {
            const double a     = t.alpha[k];
            const double ratio = t.T[k + 1] / t.T[k];
            const double pref  = (t.L_ref / t.T_ref) * t.T[k] / t.L[k];
            const double seg   = (fabs(1.0 - a) > 1e-6) ? pref / (1.0 - a) * (pow(ratio, 1.0 - a) - 1.0)
                                                       : pref * log(ratio);
            t.Y[k] = t.Y[k + 1] + seg;
        }
    }

#endif // COOLING

} // namespace astro
