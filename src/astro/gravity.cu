/* static analytic gravity: NFW + Hernquist + point-mass SMBH, applied as a Strang-split kick */
#include "../global/allvars.h"
#include "../io/input.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "astro_constants.h"
#include "gravity.h"
#include <cmath>

namespace astro {

#ifdef GRAVITY_ENABLED

    // forward declarations
    HD double     gravity_magnitude(double r, const GravityParams& p);
    HD POINT_TYPE gravity_accel(double3 pos, const GravityParams& p);
    HD void       gravity_kick_cell(hsize_t i, const VMesh* mesh, hydro::primvars* primvar, GravityParams p, double dt_half);
#ifndef CPU_DEBUG
    GLOBAL void
    kernel_gravity_kick(hsize_t n_hydro, const VMesh* mesh, hydro::primvars* primvar, GravityParams p, double dt_half);
#endif

    static GravityParams g_grav;

    // ============================================================
    // Setup
    // ============================================================

    void gravity_init() {
        g_grav.cx      = 0.5;
        g_grav.cy      = 0.5;
        g_grav.cz      = 0.5;
        const double G = units.G_in_code_units();

#ifdef NFW
        {
            const double M     = input.getParameterDouble("M_NFW") * SOLAR_MASS_G / units.UnitMass_in_g;
            const double c     = input.getParameterDouble("c_NFW");
            const double H0     = input.getParameterDouble("H0") * KM_S_IN_CGS / MPC_IN_CM * units.UnitTime_in_s();
            const double mc    = log(1.0 + c) - c / (1.0 + c);
            const double rho_s = 200.0 * c * c * c * H0 * H0 / (8.0 * PI * G * mc); // characteristic density
            g_grav.nfw_Rs      = cbrt(M / (4.0 * PI * rho_s * mc));
            g_grav.nfw_A       = G * M / mc;
        }
#endif
#ifdef HERNQUIST
        {
            const double M = input.getParameterDouble("M_BCG") * SOLAR_MASS_G / units.UnitMass_in_g;
            g_grav.hq_R    = input.getParameterDouble("R_BCG") * KPC_IN_CM / units.UnitLength_in_cm;
            g_grav.hq_GM   = G * M;
        }
#endif
#ifdef SMBH
        {
            const double M   = input.getParameterDouble("M_BH") * SOLAR_MASS_G / units.UnitMass_in_g;
            const double eps = input.getParameterDouble("smbh_softening") * KPC_IN_CM / units.UnitLength_in_cm;
            g_grav.bh_GM     = G * M;
            g_grav.bh_eps2   = eps * eps;
        }
#endif

        logging::root() << "GRAVITY: static potential enabled" << std::endl;
    }

    // ============================================================
    // Apply
    // ============================================================

    void gravity_apply(double dt_half) {
        PROFILE("GRAVITY");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;

#ifndef CPU_DEBUG
        const int tpb    = _HYDRO_BLOCK_SIZE_;
        const int blocks = ((int)mesh->n_hydro + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("GRAVITY_KICK");
            kernel_gravity_kick<<<blocks, tpb>>>(mesh->n_hydro, mesh, primvar, g_grav, dt_half);
            GPU_SYNC();
        }
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            gravity_kick_cell(i, mesh, primvar, g_grav, dt_half);
        }
#endif
    }

#ifndef CPU_DEBUG
    GLOBAL void
    kernel_gravity_kick(hsize_t n_hydro, const VMesh* mesh, hydro::primvars* primvar, GravityParams p, double dt_half) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        gravity_kick_cell(i, mesh, primvar, p, dt_half);
    }
#endif

    // ============================================================
    // Per-cell work
    // ============================================================

    // kick: v += a*dt_half; update E by the kinetic-energy change so internal energy is untouched
    HD void
    gravity_kick_cell(hsize_t i, const VMesh* mesh, hydro::primvars* primvar, GravityParams p, double dt_half) {
        const POINT_TYPE a   = gravity_accel(mesh->seeds[i], p);
        const double     rho = primvar->rho[i];
        POINT_TYPE       v   = primvar->v[i];

        double v2_old = v.x * v.x + v.y * v.y;
#ifdef dim_3D
        v2_old += v.z * v.z;
#endif
        v.x += a.x * dt_half;
        v.y += a.y * dt_half;
#ifdef dim_3D
        v.z += a.z * dt_half;
#endif
        double v2_new = v.x * v.x + v.y * v.y;
#ifdef dim_3D
        v2_new += v.z * v.z;
#endif
        primvar->v[i] = v;
        primvar->E[i] += 0.5 * rho * (v2_new - v2_old);
    }

    // radial acceleration magnitude at radius r, summed over the enabled potentials
    HD double gravity_magnitude(double r, const GravityParams& p) {
        const double r2 = r * r;
        double       g  = 0.0;
#ifdef NFW
        {
            const double x = r / p.nfw_Rs;
            g += p.nfw_A / r2 * (log(1.0 + x) - x / (1.0 + x));
        }
#endif
#ifdef HERNQUIST
        {
            const double rr = r + p.hq_R;
            g += p.hq_GM / (rr * rr);
        }
#endif
#ifdef SMBH
        {
            const double s = r2 + p.bh_eps2;
            g += p.bh_GM * r / (s * sqrt(s));
        }
#endif
        return g;
    }

    // acceleration vector at a cell position, pointing toward the center (code units, no wrap)
    HD POINT_TYPE gravity_accel(double3 pos, const GravityParams& p) {
        const double dx = p.cx - pos.x;
        const double dy = p.cy - pos.y;
#ifdef dim_3D
        const double dz = p.cz - pos.z;
        const double r  = sqrt(dx * dx + dy * dy + dz * dz);
#else
        const double r = sqrt(dx * dx + dy * dy);
#endif
        POINT_TYPE a;
        if (r < 1e-9) { // at the center there is no preferred direction
            a.x = 0.0;
            a.y = 0.0;
#ifdef dim_3D
            a.z = 0.0;
#endif
            return a;
        }
        const double g     = gravity_magnitude(r, p);
        const double inv_r = 1.0 / r;
        a.x                = g * dx * inv_r;
        a.y                = g * dy * inv_r;
#ifdef dim_3D
        a.z = g * dz * inv_r;
#endif
        return a;
    }

#endif // GRAVITY_ENABLED

} // namespace astro
