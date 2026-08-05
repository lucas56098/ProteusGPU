/* AGN feedback: chaotic-cold-accretion trigger (global reduction) + thermal deposition */
#include "../global/allvars.h"
#include "../io/input.h"
#include "../mpi/halo.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "astro_constants.h"
#include "agn.h"
#include <cmath>

namespace astro {

#ifdef AGN_ENABLED

    // forward declarations
    HD double cold_mass_contrib(hsize_t i, const VMesh* mesh, const hydro::primvars* primvar, const AgnParams& p);
    HD void   agn_deposit_cell(hsize_t          i,
                               const VMesh*     mesh,
                               hydro::primvars* primvar,
                               AgnParams        p,
                               double           f_drain,
                               double           de,
                               double           dm,
                               double           dm_jet);
#ifndef CPU_DEBUG
    GLOBAL void kernel_cold_mass(hsize_t n_hydro, const VMesh* mesh, const hydro::primvars* primvar, AgnParams p, double* acc);
    GLOBAL void kernel_agn_deposit(hsize_t          n_hydro,
                                   const VMesh*     mesh,
                                   hydro::primvars* primvar,
                                   AgnParams        p,
                                   double           f_drain,
                                   double           de,
                                   double           dm,
                                   double           dm_jet);
#endif

    static AgnParams g_agn;
    static double*   g_mcold         = nullptr; // device-visible accumulator for the cold-mass reduction
    static double    s_m_cold_cached = 0.0;     // last agn_prepare() result; reused by CFL and both agn_apply halves

    const AgnParams& agn_params() { return g_agn; }
    double           agn_m_cold_cached() { return s_m_cold_cached; }
    bool             agn_is_firing() { return s_m_cold_cached > 0.0; }

    // ============================================================
    // Setup
    // ============================================================

    void agn_init() {
        g_agn.cx = 0.5;
        g_agn.cy = 0.5;
        g_agn.cz = 0.5;

        const double R_acc = input.getParameterDouble("R_acc") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.r_acc2       = R_acc * R_acc;
        g_agn.T_cold_acc   = input.getParameterDouble("T_cold_acc");
        g_agn.C_T          = (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * PROTONMASS *
                    units.UnitVelocity_in_cm_per_s * units.UnitVelocity_in_cm_per_s / BOLTZMANN;
        g_agn.t_acc = input.getParameterDouble("t_acc") * 1.0e6 * YEAR_IN_S / units.UnitTime_in_s(); // Myr -> code
        g_agn.eta   = input.getParameterDouble("eta_agn");
        const double c_code = SPEED_OF_LIGHT / units.UnitVelocity_in_cm_per_s;
        g_agn.c2            = c_code * c_code;

        const double R_T = input.getParameterDouble("R_T") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.r_T2       = R_T * R_T;
#ifdef dim_2D
        g_agn.inv_V_T = 1.0 / (PI * R_T * R_T);
#else
        g_agn.inv_V_T = 1.0 / (4.0 / 3.0 * PI * R_T * R_T * R_T);
#endif
        g_agn.f_T   = input.getParameterDouble("f_T");
        g_agn.T_max = 5.0e9; // paper temperature limit
        // c_s^2 = gamma*(gamma-1)*e_int/rho, and T = C_T*e_int/rho, so cs2 at T_max is:
        g_agn.cs2_max = gamma_eos * (gamma_eos - 1.0) * g_agn.T_max / g_agn.C_T;

#ifdef AGN_KINETIC
        g_agn.f_K          = input.getParameterDouble("f_K");
        const double r_jet = input.getParameterDouble("R_jet") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.r_jet2       = r_jet * r_jet;
        g_agn.h_jet        = input.getParameterDouble("h_jet") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.L_jet        = input.getParameterDouble("L_jet") * KPC_IN_CM / units.UnitLength_in_cm;
#ifdef dim_2D
        g_agn.inv_Vjet = 1.0 / (2.0 * r_jet * g_agn.h_jet); // one zone: 2*r_jet (perp) x h_jet
#else
        g_agn.inv_Vjet = 1.0 / (PI * r_jet * r_jet * g_agn.h_jet); // one cylinder
#endif
        const double c_c = SPEED_OF_LIGHT / units.UnitVelocity_in_cm_per_s;
        g_agn.v_jet      = sqrt(2.0 * g_agn.eta) * c_c;
        g_agn.v_cap      = input.getParameterDouble("v_cap") * c_c;
#endif

        g_mcold = gpu_alloc<double>(1);

        logging::root() << "AGN: cold-accretion trigger (R_acc=" << R_acc << " code)"
#ifdef AGN_THERMAL
                        << ", thermal f_T=" << g_agn.f_T
#endif
#ifdef AGN_KINETIC
                        << ", kinetic f_K=" << g_agn.f_K << " v_jet=" << g_agn.v_jet
#endif
                        << " enabled" << std::endl;
    }

    // ============================================================
    // Apply — one global reduction, then local drain + deposition
    // ============================================================

    // one global reduction per step. Result is cached and reused by calc_timestep (for the AGN CFL
    // tightening) and by both Strang-split agn_apply halves. Between the two halves the state evolves,
    // but m_cold drifts by <1%/step in typical runs, so the caching error is negligible; the gain is
    // that dt is already small enough to keep the following flux step CFL-safe.
    void agn_prepare() {
        PROFILE("AGN_PREPARE");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;
        const hsize_t    n       = mesh->n_hydro;

        double m_local = 0.0;
#ifndef CPU_DEBUG
        *g_mcold = 0.0;
        const int tpb    = _HYDRO_BLOCK_SIZE_;
        const int blocks = ((int)n + tpb - 1) / tpb;
        {
            PROFILE_KERNEL("AGN_COLDMASS");
            kernel_cold_mass<<<blocks, tpb>>>(n, mesh, primvar, g_agn, g_mcold);
            GPU_SYNC();
        }
        m_local = *g_mcold;
#else
#ifdef USE_OPENMP
#pragma omp parallel for reduction(+ : m_local) schedule(static)
#endif
        for (hsize_t i = 0; i < n; i++) {
            m_local += cold_mass_contrib(i, mesh, primvar, g_agn);
        }
#endif

        double m_cold = m_local;
        proteus_mpi::halo_sum_allreduce(&m_cold);
        s_m_cold_cached = m_cold;
    }

    void agn_apply(double dt_half) {
        PROFILE("AGN");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;
        const hsize_t    n       = mesh->n_hydro;

        // read cached cold mass (populated by agn_prepare at the top of the step)
        const double m_cold = s_m_cold_cached;

        // accretion rate + feedback power
        const double Mdot = m_cold / g_agn.t_acc;
        const double Edot = g_agn.eta * Mdot * g_agn.c2;

        // per-step AGN history in physical units (rank 0, once per full step; logged even when idle so
        // the series carries the zeros). grep "AGN_POWER" from the run log -> plot like paper Fig 1.
        static int s_last_logged = -1;
        if (sim.step != s_last_logged) {
            s_last_logged      = sim.step;
            const double m2s   = units.UnitMass_in_g / SOLAR_MASS_G;                     // code mass -> Msun
            const double inv_t = 1.0 / units.UnitTime_in_s();                            // 1 / code-time-in-s
            const double e2erg = units.UnitMass_in_g * units.UnitVelocity_in_cm_per_s *  // code energy -> erg
                                 units.UnitVelocity_in_cm_per_s;
            logging::root() << "AGN_POWER: t=" << sim.t_sim << " Mcold_Msun=" << (m_cold * m2s)
                            << " Mdot_Msun_per_yr=" << (Mdot * m2s * YEAR_IN_S * inv_t)
                            << " Edot_erg_per_s=" << (Edot * e2erg * inv_t) << std::endl;
        }

        if (m_cold <= 0.0) return; // no cold gas -> nothing to drain/deposit

        // per-step deposition increments
        const double f_drain = fmin(dt_half / g_agn.t_acc, 1.0);
        double       de = 0.0, dm = 0.0, dm_jet = 0.0;
#ifdef AGN_THERMAL
        de = g_agn.f_T * Edot * g_agn.inv_V_T * dt_half; // energy density into each R_T cell
        dm = g_agn.f_T * Mdot * g_agn.inv_V_T * dt_half; // mass density into each R_T cell
#endif
#ifdef AGN_KINETIC
        // mass density loaded into ONE launch zone per step ((1-eta) f_K Mdot split over 2 zones)
        dm_jet = 0.5 * (1.0 - g_agn.eta) * g_agn.f_K * Mdot * g_agn.inv_Vjet * dt_half;
#endif

        // drain accreted cold gas + deposit thermal feedback (rank-local)
#ifndef CPU_DEBUG
        {
            PROFILE_KERNEL("AGN_DEPOSIT");
            const int tpb    = _HYDRO_BLOCK_SIZE_;
            const int blocks = ((int)n + tpb - 1) / tpb;
            kernel_agn_deposit<<<blocks, tpb>>>(n, mesh, primvar, g_agn, f_drain, de, dm, dm_jet);
            GPU_SYNC();
        }
#else
#ifdef USE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (hsize_t i = 0; i < n; i++) {
            agn_deposit_cell(i, mesh, primvar, g_agn, f_drain, de, dm, dm_jet);
        }
#endif
    }

#ifndef CPU_DEBUG
    GLOBAL void
    kernel_cold_mass(hsize_t n_hydro, const VMesh* mesh, const hydro::primvars* primvar, AgnParams p, double* acc) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        const double c = cold_mass_contrib(i, mesh, primvar, p);
        if (c > 0.0) atomicAdd(acc, c);
    }

    GLOBAL void kernel_agn_deposit(hsize_t          n_hydro,
                                   const VMesh*     mesh,
                                   hydro::primvars* primvar,
                                   AgnParams        p,
                                   double           f_drain,
                                   double           de,
                                   double           dm,
                                   double           dm_jet) {
        hsize_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_hydro) return;
        agn_deposit_cell(i, mesh, primvar, p, f_drain, de, dm, dm_jet);
    }
#endif

    // ============================================================
    // Per-cell work
    // ============================================================

    // cell mass if it is cold gas inside the accretion radius, else 0
    HD double cold_mass_contrib(hsize_t i, const VMesh* mesh, const hydro::primvars* primvar, const AgnParams& p) {
        const double dx = mesh->seeds[i].x - p.cx;
        const double dy = mesh->seeds[i].y - p.cy;
#ifdef dim_3D
        const double dz = mesh->seeds[i].z - p.cz;
        const double r2 = dx * dx + dy * dy + dz * dz;
#else
        const double r2 = dx * dx + dy * dy;
#endif
        if (r2 >= p.r_acc2) return 0.0;

        const double rho = primvar->rho[i];
        POINT_TYPE   v   = primvar->v[i];
        double       v2  = v.x * v.x + v.y * v.y;
#ifdef dim_3D
        v2 += v.z * v.z;
#endif
        const double e_int = primvar->E[i] - 0.5 * rho * v2;
        if (e_int <= 0.0) return 0.0;
        if (p.C_T * e_int / rho >= p.T_cold_acc) return 0.0; // not cold
        return rho * mesh->volumes[i];
    }

    // drain accreted cold gas (R_acc), deposit thermal energy+mass (R_T), and load the kinetic jets
    HD void agn_deposit_cell(hsize_t          i,
                             const VMesh*     mesh,
                             hydro::primvars* primvar,
                             AgnParams        p,
                             double           f_drain,
                             double           de,
                             double           dm,
                             double           dm_jet) {
        const double dx = mesh->seeds[i].x - p.cx;
        const double dy = mesh->seeds[i].y - p.cy;
#ifdef dim_3D
        const double dz = mesh->seeds[i].z - p.cz;
        const double r2 = dx * dx + dy * dy + dz * dz;
#else
        const double r2 = dx * dx + dy * dy;
#endif
        (void)dm_jet;
        double rho = primvar->rho[i];

        // deplete accreted cold gas (removes mass and its energy, leaving v and T unchanged)
        if (r2 < p.r_acc2) {
            POINT_TYPE v  = primvar->v[i];
            double     v2 = v.x * v.x + v.y * v.y;
#ifdef dim_3D
            v2 += v.z * v.z;
#endif
            const double e_int = primvar->E[i] - 0.5 * rho * v2;
            if (e_int > 0.0 && p.C_T * e_int / rho < p.T_cold_acc) {
                rho *= (1.0 - f_drain);
                primvar->rho[i] = rho;
                primvar->E[i] *= (1.0 - f_drain);
            }
        }

#ifdef AGN_THERMAL
        // uniform thermal deposition inside R_T
        if (r2 < p.r_T2) {
            if (dm > 0.0) {
                primvar->E[i] *= (1.0 + dm / rho); // added mass at cell v, T
                rho += dm;
                primvar->rho[i] = rho;
            }
            primvar->E[i] += de;

            // temperature cap
            POINT_TYPE v  = primvar->v[i];
            double     v2 = v.x * v.x + v.y * v.y;
#ifdef dim_3D
            v2 += v.z * v.z;
#endif
            const double e_max = rho * p.T_max / p.C_T;
            const double E_max = 0.5 * rho * v2 + e_max;
            if (primvar->E[i] > E_max) primvar->E[i] = E_max;
        }
#endif

#ifdef AGN_KINETIC
        // load mass + outward momentum into the two launch zones on the +/-y axis
#ifdef dim_3D
        const double perp2 = dx * dx + dz * dz;
#else
        const double perp2 = dx * dx;
#endif
        const double ady = fabs(dy);
        if (dm_jet > 0.0 && perp2 < p.r_jet2 && ady > p.L_jet && ady < p.L_jet + p.h_jet) {
            const double sign  = (dy > 0.0) ? 1.0 : -1.0; // momentum points away from center
            POINT_TYPE   v     = primvar->v[i];
            const double rho_new = rho + dm_jet;
            // conserve momentum + add the KE carried by the injected slug (E += 1/2 dm v_jet^2).
            // The natural inelastic mixing heat (slug decelerates against background mass) ends up in
            // e_int automatically. If we only rebuilt E from the mass-weighted v_new, we would deliver
            // only a fraction dm/(rho+dm) of the intended f_K*Edot -- see paper Eq. 15.
            v.x = (rho * v.x) / rho_new;
            v.y = (rho * v.y + dm_jet * sign * p.v_jet) / rho_new;
#ifdef dim_3D
            v.z = (rho * v.z) / rho_new;
#endif
            // velocity cap here is a per-step numerical safety inside the launch zone; the global
            // r<R_lim cap (astro::limiters) is what enforces the paper's <0.05c constraint everywhere
            // in the central region.
            double v2new = v.x * v.x + v.y * v.y;
#ifdef dim_3D
            v2new += v.z * v.z;
#endif
            const double vmag = sqrt(v2new);
            if (vmag > p.v_cap) {
                const double s = p.v_cap / vmag;
                v.x *= s;
                v.y *= s;
#ifdef dim_3D
                v.z *= s;
#endif
            }
            primvar->v[i]   = v;
            primvar->rho[i] = rho_new;
            primvar->E[i] += 0.5 * dm_jet * p.v_jet * p.v_jet;
        }
#endif
    }

#endif // AGN_ENABLED

} // namespace astro
