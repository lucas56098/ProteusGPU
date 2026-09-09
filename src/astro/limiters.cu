/* central-region hard clamps: T < T_max and |v| < v_cap inside r < R_lim (paper Sect. 2.1) */
#include "../global/allvars.h"
#include "../io/input.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "astro_constants.h"
#include "limiters.h"
#include <cmath>

namespace astro {

#ifdef LIMITERS

    // forward declarations
    HD void limiter_cell(hsize_t i, const VMesh* mesh, hydro::primvars* primvar, LimiterParams p);

    static LimiterParams g_lim;

    const LimiterParams& limiters_params() {
        return g_lim;
    }

    // ============================================================
    // Setup
    // ============================================================

    void limiters_init() {
        g_lim.cx = 0.5;
        g_lim.cy = 0.5;
        g_lim.cz = 0.5;

        const double R_lim = input.getParameterDouble("R_lim") * KPC_IN_CM / units.UnitLength_in_cm;
        g_lim.r_lim2       = R_lim * R_lim;
        g_lim.T_max        = input.getParameterDouble("T_max_lim");
        g_lim.C_T          = (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * PROTONMASS * units.UnitVelocity_in_cm_per_s *
                    units.UnitVelocity_in_cm_per_s / BOLTZMANN;
        g_lim.e_max_c       = g_lim.T_max / g_lim.C_T; // e_int/rho ceiling
        const double c_code = SPEED_OF_LIGHT / units.UnitVelocity_in_cm_per_s;
        g_lim.v_cap         = input.getParameterDouble("v_cap_lim") * c_code;
        g_lim.v_cap2        = g_lim.v_cap * g_lim.v_cap;

        logging::root() << "LIMITERS: r<" << R_lim << " code, T<" << g_lim.T_max << " K, |v|<" << g_lim.v_cap
                        << " code enabled" << std::endl;
    }

    // ============================================================
    // Apply
    // ============================================================

    void limiters_apply() {
        PROFILE("LIMITERS");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;

        const LimiterParams p = g_lim;

        parallel_for<_HYDRO_BLOCK_SIZE_>(
            "LIMITERS_KERNEL", mesh->n_hydro, [=] HD(size_t i) { limiter_cell(i, mesh, primvar, p); });
    }

    // ============================================================
    // Per-cell work
    // ============================================================

    // clamp T (via internal energy) and |v| (via momentum) inside r < R_lim
    HD void limiter_cell(hsize_t i, const VMesh* mesh, hydro::primvars* primvar, LimiterParams p) {
        const double dx = mesh->seeds[i].x - p.cx;
        const double dy = mesh->seeds[i].y - p.cy;
#ifdef dim_3D
        const double dz = mesh->seeds[i].z - p.cz;
        const double r2 = dx * dx + dy * dy + dz * dz;
#else
        const double r2 = dx * dx + dy * dy;
#endif
        if (r2 >= p.r_lim2) return;

        const double rho = primvar->rho[i];
        POINT_TYPE   v   = primvar->v[i];

        // |v| cap: scale v uniformly; the shed kinetic energy is left in E and shows up as heat
        // (subject to the T cap below).
        double v2 = v.x * v.x + v.y * v.y;
#ifdef dim_3D
        v2 += v.z * v.z;
#endif
        if (v2 > p.v_cap2) {
            const double s = sqrt(p.v_cap2 / v2);
            v.x *= s;
            v.y *= s;
#ifdef dim_3D
            v.z *= s;
#endif
            primvar->v[i] = v;
            v2            = p.v_cap2;
        }

        // T cap: bound e_int = E - 1/2 rho v^2 by rho * (T_max / C_T)
        const double e_int_max = rho * p.e_max_c;
        const double E_max     = 0.5 * rho * v2 + e_int_max;
        if (primvar->E[i] > E_max) primvar->E[i] = E_max;
    }

#endif // LIMITERS

} // namespace astro
