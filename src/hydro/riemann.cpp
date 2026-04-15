#include "riemann.h"
#include "../global/allvars.h"

namespace hydro {

    flux_t riemann_hll(prim st_l, prim st_r) {

        // calc f_l and f_r
        flux_t f_l = get_flux(&st_l);
        flux_t f_r = get_flux(&st_r);

        // cache pressure (clamp to zero for numerical safety)
        double P_l = fmax(0.0, get_P_ideal_gas(&st_l));
        double P_r = fmax(0.0, get_P_ideal_gas(&st_r));

        // wave speeds
        double SL = fmin(st_l.v.x - sqrt((gamma_eos * P_l) / st_l.rho), st_r.v.x - sqrt((gamma_eos * P_r) / st_r.rho));
        double SR = fmax(st_l.v.x + sqrt((gamma_eos * P_l) / st_l.rho), st_r.v.x + sqrt((gamma_eos * P_r) / st_r.rho));

        // calc HLL flux
        flux_t flux;

        if (SL >= 0) {
            flux = f_l;
        } else if (SL < 0 && SR > 0) {
            flux.rho = (SR * f_l.rho - SL * f_r.rho + SL * SR * (st_r.rho - st_l.rho)) / (SR - SL);
            flux.v.x =
                (SR * f_l.v.x - SL * f_r.v.x + SL * SR * (st_r.rho * st_r.v.x - st_l.rho * st_l.v.x)) / (SR - SL);
            flux.v.y =
                (SR * f_l.v.y - SL * f_r.v.y + SL * SR * (st_r.rho * st_r.v.y - st_l.rho * st_l.v.y)) / (SR - SL);
#ifdef dim_3D
            flux.v.z =
                (SR * f_l.v.z - SL * f_r.v.z + SL * SR * (st_r.rho * st_r.v.z - st_l.rho * st_l.v.z)) / (SR - SL);
#endif
            flux.E = (SR * f_l.E - SL * f_r.E + SL * SR * (st_r.E - st_l.E)) / (SR - SL);
        } else if (SR <= 0) {
            flux = f_r;
        }

        return flux;
    }

    flux_t riemann_hllc(prim st_l, prim st_r) {

        // calc f_l and f_r
        flux_t f_l = get_flux(&st_l);
        flux_t f_r = get_flux(&st_r);

        // cache pressure (clamp to zero for numerical safety)
        double P_l = fmax(0.0, get_P_ideal_gas(&st_l));
        double P_r = fmax(0.0, get_P_ideal_gas(&st_r));

        // wave speeds
        double SL = fmin(st_l.v.x - sqrt((gamma_eos * P_l) / st_l.rho), st_r.v.x - sqrt((gamma_eos * P_r) / st_r.rho));
        double SR = fmax(st_l.v.x + sqrt((gamma_eos * P_l) / st_l.rho), st_r.v.x + sqrt((gamma_eos * P_r) / st_r.rho));

        // calculate S_star
        double S_star = (P_r - P_l + st_l.rho * st_l.v.x * (SL - st_l.v.x) - st_r.rho * st_r.v.x * (SR - st_r.v.x)) /
                        (st_l.rho * (SL - st_l.v.x) - st_r.rho * (SR - st_r.v.x));

        // HLLC solver for F
        flux_t flux;
        if (0 <= SL) {
            flux = f_l;
        } else if (SL <= 0 && 0 <= S_star) {
            flux.rho = (S_star * (SL * st_l.rho - f_l.rho)) / (SL - S_star);
            flux.v.x = (S_star * (SL * st_l.rho * st_l.v.x - f_l.v.x) +
                        SL * (P_l + st_l.rho * (SL - st_l.v.x) * (S_star - st_l.v.x))) /
                       (SL - S_star);
            flux.v.y = (S_star * (SL * st_l.rho * st_l.v.y - f_l.v.y)) / (SL - S_star);
#ifdef dim_3D
            flux.v.z = (S_star * (SL * st_l.rho * st_l.v.z - f_l.v.z)) / (SL - S_star);
#endif
            flux.E = (S_star * (SL * st_l.E - f_l.E) +
                      SL * (P_l + st_l.rho * (SL - st_l.v.x) * (S_star - st_l.v.x)) * S_star) /
                     (SL - S_star);
        } else if (S_star <= 0 && 0 <= SR) {
            flux.rho = (S_star * (SR * st_r.rho - f_r.rho)) / (SR - S_star);
            flux.v.x = (S_star * (SR * st_r.rho * st_r.v.x - f_r.v.x) +
                        SR * (P_r + st_r.rho * (SR - st_r.v.x) * (S_star - st_r.v.x))) /
                       (SR - S_star);
            flux.v.y = (S_star * (SR * st_r.rho * st_r.v.y - f_r.v.y)) / (SR - S_star);
#ifdef dim_3D
            flux.v.z = (S_star * (SR * st_r.rho * st_r.v.z - f_r.v.z)) / (SR - S_star);
#endif
            flux.E = (S_star * (SR * st_r.E - f_r.E) +
                      SR * (P_r + st_r.rho * (SR - st_r.v.x) * (S_star - st_r.v.x)) * S_star) /
                     (SR - S_star);
        } else if (0 >= SR) {
            flux = f_r;
        }

        return flux;
    }

    flux_t get_flux(const prim* state) {

        double P = get_P_ideal_gas(state);

        // calc flux
        flux_t flux;

        flux.rho = state->rho * state->v.x;
        flux.v.x = state->rho * state->v.x * state->v.x + P;
        flux.v.y = state->rho * state->v.x * state->v.y;
#ifdef dim_3D
        flux.v.z = state->rho * state->v.x * state->v.z;
#endif
        flux.E = (state->E + P) * state->v.x;

        return flux;
    }

    double get_P_ideal_gas(const prim* state) {
#ifdef dim_2D
        return (gamma_eos - 1) * (state->E - (0.5 * state->rho * (state->v.x * state->v.x + state->v.y * state->v.y)));
#else
        return (gamma_eos - 1) *
               (state->E -
                (0.5 * state->rho * (state->v.x * state->v.x + state->v.y * state->v.y + state->v.z * state->v.z)));
#endif
    }

} // namespace hydro