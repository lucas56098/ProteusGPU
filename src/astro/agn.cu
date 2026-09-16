// implements the black hole feedback (agn.h)

#include "../global/allvars.h"
#include "../io/input.h"
#include "../mpi/halo.h"
#include "../profiler/profiler.h"
#include "../voronoi/voronoi.h"
#include "agn.h"
#include "astro_constants.h"
#include <cmath>

namespace astro {

#ifdef AGN_ENABLED

    HD double cold_mass_contrib(uint64_t i, const VMesh* mesh, const hydro::primvars* primvar, const AgnParams& p);
    HD void   agn_deposit_cell(uint64_t         i,
                               const VMesh*     mesh,
                               hydro::primvars* primvar,
                               AgnParams        p,
                               double           f_drain,
                               double           de,
                               double           dm,
                               double           dm_jet);
    static AgnParams g_agn;
    static double    s_m_cold_cached = 0.0;

    const AgnParams& agn_params() {
        return g_agn;
    }
    bool agn_is_firing() {
        return s_m_cold_cached > 0.0;
    }

    // reads the parameters and turns them into code units
    void agn_init() {
        g_agn.cx = 0.5;
        g_agn.cy = 0.5;
        g_agn.cz = 0.5;

        const double R_acc = input.get_parameter_double("R_acc") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.r_acc2       = R_acc * R_acc;
        g_agn.T_cold_acc   = input.get_parameter_double("T_cold_acc");
        g_agn.C_T          = (gamma_eos - 1.0) * MEAN_MOL_WEIGHT * PROTONMASS * units.UnitVelocity_in_cm_per_s *
                    units.UnitVelocity_in_cm_per_s / BOLTZMANN;
        g_agn.t_acc         = input.get_parameter_double("t_acc") * 1.0e6 * YEAR_IN_S / units.UnitTime_in_s();
        g_agn.eta           = input.get_parameter_double("eta_agn");
        const double c_code = SPEED_OF_LIGHT / units.UnitVelocity_in_cm_per_s;
        g_agn.c2            = c_code * c_code;

        const double R_T = input.get_parameter_double("R_T") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.r_T2       = R_T * R_T;
#ifdef dim_2D
        g_agn.inv_V_T = 1.0 / (PI * R_T * R_T);
#else
        g_agn.inv_V_T = 1.0 / (4.0 / 3.0 * PI * R_T * R_T * R_T);
#endif
        g_agn.f_T     = input.get_parameter_double("f_T");
        g_agn.T_max   = 5.0e9;
        g_agn.cs2_max = gamma_eos * (gamma_eos - 1.0) * g_agn.T_max / g_agn.C_T;

#ifdef AGN_KINETIC
        g_agn.f_K          = input.get_parameter_double("f_K");
        const double r_jet = input.get_parameter_double("R_jet") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.r_jet2       = r_jet * r_jet;
        g_agn.h_jet        = input.get_parameter_double("h_jet") * KPC_IN_CM / units.UnitLength_in_cm;
        g_agn.L_jet        = input.get_parameter_double("L_jet") * KPC_IN_CM / units.UnitLength_in_cm;
#ifdef dim_2D
        g_agn.inv_Vjet = 1.0 / (2.0 * r_jet * g_agn.h_jet);
#else
        g_agn.inv_Vjet = 1.0 / (PI * r_jet * r_jet * g_agn.h_jet);
#endif
        const double c_c = SPEED_OF_LIGHT / units.UnitVelocity_in_cm_per_s;
        g_agn.v_jet      = sqrt(2.0 * g_agn.eta) * c_c;
        g_agn.v_cap      = input.get_parameter_double("v_cap") * c_c;
#endif

        logging::root() << "AGN: cold-accretion trigger (R_acc=" << R_acc << " code)"
#ifdef AGN_THERMAL
                        << ", thermal f_T=" << g_agn.f_T
#endif
#ifdef AGN_KINETIC
                        << ", kinetic f_K=" << g_agn.f_K << " v_jet=" << g_agn.v_jet
#endif
                        << " enabled" << std::endl;
    }

    // cold gas mass near the centre, over all ranks; that is what the black hole feeds on this step
    void agn_prepare() {
        PROFILE("AGN_PREPARE");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;
        const uint64_t   n       = mesh->n_hydro;

        const AgnParams p       = g_agn;
        const double    m_local = parallel_reduce_sum<_HYDRO_BLOCK_SIZE_, double>(
            "AGN_COLDMASS", n, [=] HD(size_t i) { return cold_mass_contrib(i, mesh, primvar, p); });

        double m_cold = m_local;
        proteus_mpi::halo_sum_allreduce(&m_cold);
        s_m_cold_cached = m_cold;
    }

    // turns that mass into heat and jet momentum, half a step at a time
    void agn_apply(double dt_half) {
        PROFILE("AGN");
        VMesh*           mesh    = sim.mesh;
        hydro::primvars* primvar = sim.primvar;
        const uint64_t   n       = mesh->n_hydro;

        const double m_cold = s_m_cold_cached;

        // accretion rate and the power that comes with it
        const double Mdot = m_cold / g_agn.t_acc;
        const double Edot = g_agn.eta * Mdot * g_agn.c2;

        // one line per step, not per half step
        static int s_last_logged = -1;
        if (sim.step != s_last_logged) {
            s_last_logged      = sim.step;
            const double m2s   = units.UnitMass_in_g / SOLAR_MASS_G;
            const double inv_t = 1.0 / units.UnitTime_in_s();
            const double e2erg = units.UnitMass_in_g * units.UnitVelocity_in_cm_per_s * units.UnitVelocity_in_cm_per_s;
            logging::root() << "AGN_POWER: t=" << sim.t_sim << " Mcold_Msun=" << (m_cold * m2s)
                            << " Mdot_Msun_per_yr=" << (Mdot * m2s * YEAR_IN_S * inv_t)
                            << " Edot_erg_per_s=" << (Edot * e2erg * inv_t) << std::endl;
        }

        if (m_cold <= 0.0) return;

        // share of the cold gas that leaves the accretion region in this half step
        const double f_drain = fmin(dt_half / g_agn.t_acc, 1.0);
        // and what the thermal and the kinetic part put back, per volume
        double de = 0.0, dm = 0.0, dm_jet = 0.0;
#ifdef AGN_THERMAL
        de = g_agn.f_T * Edot * g_agn.inv_V_T * dt_half;
        dm = g_agn.f_T * Mdot * g_agn.inv_V_T * dt_half;
#endif
#ifdef AGN_KINETIC
        dm_jet = 0.5 * (1.0 - g_agn.eta) * g_agn.f_K * Mdot * g_agn.inv_Vjet * dt_half;
#endif

        const AgnParams p = g_agn;

        parallel_for<_HYDRO_BLOCK_SIZE_>(
            "AGN_DEPOSIT", n, [=] HD(size_t i) { agn_deposit_cell(i, mesh, primvar, p, f_drain, de, dm, dm_jet); });
    }

    // mass of this cell if it is near the centre and cold, else nothing
    HD double cold_mass_contrib(uint64_t i, const VMesh* mesh, const hydro::primvars* primvar, const AgnParams& p) {
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
        if (p.C_T * e_int / rho >= p.T_cold_acc) return 0.0;
        return rho * mesh->volumes[i];
    }

    // one cell: drain, heat, and push if it lies in a jet
    HD void agn_deposit_cell(uint64_t         i,
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
        (void)de;
        (void)dm;
        double rho = primvar->rho[i];

        // cold gas near the centre loses the share that fell in
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
        // heat and mass, spread evenly over the sphere, capped at T_max
        if (r2 < p.r_T2) {
            if (dm > 0.0) {
                primvar->E[i] *= (1.0 + dm / rho);
                rho += dm;
                primvar->rho[i] = rho;
            }
            primvar->E[i] += de;

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
#ifdef dim_3D
        const double perp2 = dx * dx + dz * dz;
#else
        const double perp2 = dx * dx;
#endif
        const double ady = fabs(dy);
        // inside a jet: add mass at jet speed, away from the centre
        if (dm_jet > 0.0 && perp2 < p.r_jet2 && ady > p.L_jet && ady < p.L_jet + p.h_jet) {
            const double sign    = (dy > 0.0) ? 1.0 : -1.0;
            POINT_TYPE   v       = primvar->v[i];
            const double rho_new = rho + dm_jet;
            v.x                  = (rho * v.x) / rho_new;
            v.y                  = (rho * v.y + dm_jet * sign * p.v_jet) / rho_new;
#ifdef dim_3D
            v.z = (rho * v.z) / rho_new;
#endif
            double v2new = v.x * v.x + v.y * v.y;
#ifdef dim_3D
            v2new += v.z * v.z;
#endif
            const double vmag = sqrt(v2new);
            // a light cell must not run off
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

#endif

} // namespace astro
