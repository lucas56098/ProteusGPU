// implements the stellar source terms (stars.h)

#include "../global/allvars.h"
#include "../io/input.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "astro_constants.h"
#include "stars.h"
#include <cmath>

namespace astro {

#ifdef SF_FEEDBACK

    HD void stars_source_cell(uint64_t i, const VMesh* mesh, hydro::primvars* primvar, StarParams p, double dt_half);

    static StarParams g_star;

    // parameters of the galaxy and of the star formation, in code units
    void stars_init() {
        g_star.cx      = 0.5;
        g_star.cy      = 0.5;
        g_star.cz      = 0.5;
        const double G = units.G_in_code_units();

        const double M_BCG     = input.get_parameter_double("M_BCG") * SOLAR_MASS_G / units.UnitMass_in_g;
        const double R_BCG     = input.get_parameter_double("R_BCG") * KPC_IN_CM / units.UnitLength_in_cm;
        const double Gamma     = input.get_parameter_double("Gamma_SNIa");
        const double E_SN      = input.get_parameter_double("E_SNIa");
        const double alpha     = input.get_parameter_double("alpha_SNIa");
        const double Gamma_cgs = Gamma / (YEAR_IN_S * SOLAR_MASS_G);

        g_star.snia_Ce = Gamma_cgs * E_SN * units.UnitTime_in_s() /
                         (units.UnitVelocity_in_cm_per_s * units.UnitVelocity_in_cm_per_s);
        g_star.snia_Cm   = alpha * units.UnitTime_in_s();
        g_star.bcg_norm  = M_BCG * R_BCG / (2.0 * PI);
        g_star.bcg_R     = R_BCG;
        g_star.bcg_rsoft = 0.1 * R_BCG;

        g_star.sf_eff        = input.get_parameter_double("eff_SF");
        const double c_code  = SPEED_OF_LIGHT / units.UnitVelocity_in_cm_per_s;
        g_star.sf_c2         = c_code * c_code;
        g_star.sf_G          = G;
        const double n_SF    = input.get_parameter_double("n_SF");
        g_star.sf_rho_thresh = n_SF * PROTONMASS / (HYDROGEN_MASSFRAC * units.UnitDensity_in_cgs());
        g_star.sf_T_max      = input.get_parameter_double("T_SF");
        g_star.sf_C_T        = (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * PROTONMASS * units.UnitVelocity_in_cm_per_s *
                        units.UnitVelocity_in_cm_per_s / BOLTZMANN;
        const double R_acc = input.get_parameter_double("R_acc") * KPC_IN_CM / units.UnitLength_in_cm;
        const double R_SF  = input.get_parameter_double("R_SF") * KPC_IN_CM / units.UnitLength_in_cm;
        g_star.sf_r_in2    = R_acc * R_acc;
        g_star.sf_r_out2   = R_SF * R_SF;

        logging::root() << "STARS: SNIa + particle-free SF feedback enabled" << std::endl;
    }

    void stars_apply(double dt_half) {
        PROFILE("STARS");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;

        const StarParams p = g_star;

        parallel_for<_HYDRO_BLOCK_SIZE_>(
            "STARS_KERNEL", mesh->n_hydro, [=] HD(size_t i) { stars_source_cell(i, mesh, primvar, p, dt_half); });
    }

    // one cell: mass and heat from the old stars, then the star formation feedback
    HD void stars_source_cell(uint64_t i, const VMesh* mesh, hydro::primvars* primvar, StarParams p, double dt_half) {
        const double dx = mesh->seeds[i].x - p.cx;
        const double dy = mesh->seeds[i].y - p.cy;
#ifdef dim_3D
        const double dz = mesh->seeds[i].z - p.cz;
        const double r2 = dx * dx + dy * dy + dz * dz;
#else
        const double r2 = dx * dx + dy * dy;
#endif
        const double r   = sqrt(r2);
        double       rho = primvar->rho[i];

        // stellar density here, softened at the centre
        const double rs      = fmax(r, p.bcg_rsoft);
        const double rr      = rs + p.bcg_R;
        const double rho_bcg = p.bcg_norm / (rs * rr * rr * rr);
        const double dm      = p.snia_Cm * rho_bcg * dt_half;
        // the gas the stars give back arrives at the velocity of the cell
        if (dm > 0.0) {
            primvar->E[i] *= (1.0 + dm / rho);
            rho += dm;
            primvar->rho[i] = rho;
        }
        primvar->E[i] += p.snia_Ce * rho_bcg * dt_half;

        // dense, cold gas in the shell turns part of its rest mass into heat, over a free fall time
        if (r2 > p.sf_r_in2 && r2 < p.sf_r_out2 && rho > p.sf_rho_thresh) {
            POINT_TYPE v  = primvar->v[i];
            double     v2 = v.x * v.x + v.y * v.y;
#ifdef dim_3D
            v2 += v.z * v.z;
#endif
            const double e_int = primvar->E[i] - 0.5 * rho * v2;
            if (e_int > 0.0 && p.sf_C_T * e_int / rho < p.sf_T_max) {
                const double t_ff = sqrt(3.0 * PI / (32.0 * p.sf_G * rho));
                primvar->E[i] += p.sf_eff * rho * p.sf_c2 * dt_half / t_ff;
            }
        }
    }

#endif

} // namespace astro
