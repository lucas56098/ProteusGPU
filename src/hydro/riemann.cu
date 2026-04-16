#include "../global/allvars.h"
#include "riemann.h"

namespace hydro {

    HD flux_t riemann_hll(prim st_l, prim st_r) {

        // compute pressure once per side
        double v2_l = st_l.v.x * st_l.v.x + st_l.v.y * st_l.v.y;
        double v2_r = st_r.v.x * st_r.v.x + st_r.v.y * st_r.v.y;
#ifdef dim_3D
        v2_l += st_l.v.z * st_l.v.z;
        v2_r += st_r.v.z * st_r.v.z;
#endif
        double P_l = fmax(0.0, (gamma_eos - 1.0) * (st_l.E - 0.5 * st_l.rho * v2_l));
        double P_r = fmax(0.0, (gamma_eos - 1.0) * (st_r.E - 0.5 * st_r.rho * v2_r));

        // inline flux computation
        double rho_vx_l = st_l.rho * st_l.v.x;
        double rho_vx_r = st_r.rho * st_r.v.x;

        flux_t f_l, f_r;
        f_l.rho = rho_vx_l;
        f_l.v.x = rho_vx_l * st_l.v.x + P_l;
        f_l.v.y = rho_vx_l * st_l.v.y;
        f_l.E   = (st_l.E + P_l) * st_l.v.x;

        f_r.rho = rho_vx_r;
        f_r.v.x = rho_vx_r * st_r.v.x + P_r;
        f_r.v.y = rho_vx_r * st_r.v.y;
        f_r.E   = (st_r.E + P_r) * st_r.v.x;
#ifdef dim_3D
        f_l.v.z = rho_vx_l * st_l.v.z;
        f_r.v.z = rho_vx_r * st_r.v.z;
#endif

        // wave speeds
        double c_l = sqrt(gamma_eos * P_l / st_l.rho);
        double c_r = sqrt(gamma_eos * P_r / st_r.rho);
        double SL  = fmin(st_l.v.x - c_l, st_r.v.x - c_r);
        double SR  = fmax(st_l.v.x + c_l, st_r.v.x + c_r);

        // calc HLL flux
        flux_t flux;
        if (SL >= 0) {
            flux = f_l;
        } else if (SR > 0) {
            double inv  = 1.0 / (SR - SL);
            double SLSR = SL * SR;
            flux.rho    = (SR * f_l.rho - SL * f_r.rho + SLSR * (st_r.rho - st_l.rho)) * inv;
            flux.v.x    = (SR * f_l.v.x - SL * f_r.v.x + SLSR * (rho_vx_r - rho_vx_l)) * inv;
            flux.v.y    = (SR * f_l.v.y - SL * f_r.v.y + SLSR * (st_r.rho * st_r.v.y - st_l.rho * st_l.v.y)) * inv;
#ifdef dim_3D
            flux.v.z = (SR * f_l.v.z - SL * f_r.v.z + SLSR * (st_r.rho * st_r.v.z - st_l.rho * st_l.v.z)) * inv;
#endif
            flux.E = (SR * f_l.E - SL * f_r.E + SLSR * (st_r.E - st_l.E)) * inv;
        } else {
            flux = f_r;
        }

        return flux;
    }

    HD flux_t riemann_hllc(prim st_l, prim st_r) {

        // compute pressure once per side (avoids triple-computation via get_flux→get_P)
        double v2_l = st_l.v.x * st_l.v.x + st_l.v.y * st_l.v.y;
        double v2_r = st_r.v.x * st_r.v.x + st_r.v.y * st_r.v.y;
#ifdef dim_3D
        v2_l += st_l.v.z * st_l.v.z;
        v2_r += st_r.v.z * st_r.v.z;
#endif
        double P_l = fmax(0.0, (gamma_eos - 1.0) * (st_l.E - 0.5 * st_l.rho * v2_l));
        double P_r = fmax(0.0, (gamma_eos - 1.0) * (st_r.E - 0.5 * st_r.rho * v2_r));

        // inline flux computation (reuses P_l, P_r)
        double rho_vx_l = st_l.rho * st_l.v.x;
        double rho_vx_r = st_r.rho * st_r.v.x;

        flux_t f_l, f_r;
        f_l.rho = rho_vx_l;
        f_l.v.x = rho_vx_l * st_l.v.x + P_l;
        f_l.v.y = rho_vx_l * st_l.v.y;
        f_l.E   = (st_l.E + P_l) * st_l.v.x;

        f_r.rho = rho_vx_r;
        f_r.v.x = rho_vx_r * st_r.v.x + P_r;
        f_r.v.y = rho_vx_r * st_r.v.y;
        f_r.E   = (st_r.E + P_r) * st_r.v.x;
#ifdef dim_3D
        f_l.v.z = rho_vx_l * st_l.v.z;
        f_r.v.z = rho_vx_r * st_r.v.z;
#endif

        // wave speeds (precompute sound speed factors)
        double c_l = sqrt(gamma_eos * P_l / st_l.rho);
        double c_r = sqrt(gamma_eos * P_r / st_r.rho);
        double SL  = fmin(st_l.v.x - c_l, st_r.v.x - c_r);
        double SR  = fmax(st_l.v.x + c_l, st_r.v.x + c_r);

        // contact speed
        double dSL = SL - st_l.v.x;
        double dSR = SR - st_r.v.x;
        double S_star =
            (P_r - P_l + st_l.rho * st_l.v.x * dSL - st_r.rho * st_r.v.x * dSR) / (st_l.rho * dSL - st_r.rho * dSR);

        // HLLC solver
        flux_t flux;
        if (0.0 <= SL) {
            flux = f_l;
        } else if (S_star >= 0.0) {
            double inv_SL_Sstar = 1.0 / (SL - S_star);
            double P_star       = P_l + st_l.rho * dSL * (S_star - st_l.v.x);
            flux.rho            = (S_star * (SL * st_l.rho - f_l.rho)) * inv_SL_Sstar;
            flux.v.x            = (S_star * (SL * rho_vx_l - f_l.v.x) + SL * P_star) * inv_SL_Sstar;
            flux.v.y            = (S_star * (SL * st_l.rho * st_l.v.y - f_l.v.y)) * inv_SL_Sstar;
#ifdef dim_3D
            flux.v.z = (S_star * (SL * st_l.rho * st_l.v.z - f_l.v.z)) * inv_SL_Sstar;
#endif
            flux.E = (S_star * (SL * st_l.E - f_l.E) + SL * P_star * S_star) * inv_SL_Sstar;
        } else if (0.0 <= SR) {
            double inv_SR_Sstar = 1.0 / (SR - S_star);
            double P_star       = P_r + st_r.rho * dSR * (S_star - st_r.v.x);
            flux.rho            = (S_star * (SR * st_r.rho - f_r.rho)) * inv_SR_Sstar;
            flux.v.x            = (S_star * (SR * rho_vx_r - f_r.v.x) + SR * P_star) * inv_SR_Sstar;
            flux.v.y            = (S_star * (SR * st_r.rho * st_r.v.y - f_r.v.y)) * inv_SR_Sstar;
#ifdef dim_3D
            flux.v.z = (S_star * (SR * st_r.rho * st_r.v.z - f_r.v.z)) * inv_SR_Sstar;
#endif
            flux.E = (S_star * (SR * st_r.E - f_r.E) + SR * P_star * S_star) * inv_SR_Sstar;
        } else {
            flux = f_r;
        }

        return flux;
    }

    HD double get_P_ideal_gas(const prim* state) {
#ifdef dim_2D
        return (gamma_eos - 1) * (state->E - (0.5 * state->rho * (state->v.x * state->v.x + state->v.y * state->v.y)));
#else
        return (gamma_eos - 1) *
               (state->E -
                (0.5 * state->rho * (state->v.x * state->v.x + state->v.y * state->v.y + state->v.z * state->v.z)));
#endif
    }

} // namespace hydro