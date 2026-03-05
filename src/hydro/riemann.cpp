#include "riemann.h"
#include "../global/allvars.h"


namespace hydro {

    prim riemann_hll(hsize_t i, hsize_t j, prim st_l, prim st_r, const VMesh* mesh) {

        // rotate state_j and state_i into frame
        double3 normal = mesh->face_normal[mesh->face_ptr[i] + j];
        geom g = compute_geom(normal);
        rotate_to_face(&st_l, &g);
        rotate_to_face(&st_r, &g);

        // calc f_l and f_r
        prim f_l = get_flux(&st_l);
        prim f_r = get_flux(&st_r);

        
        // wave speeds
        double SL = std::min(st_l.v.x - sqrt((_gamma_ * get_P_ideal_gas(&st_l))/st_l.rho), st_r.v.x - sqrt((_gamma_ * get_P_ideal_gas(&st_r))/st_r.rho));
        double SR = std::max(st_l.v.x + sqrt((_gamma_ * get_P_ideal_gas(&st_l))/st_l.rho), st_r.v.x + sqrt((_gamma_ * get_P_ideal_gas(&st_r))/st_r.rho));

        // calc HLL flux
        prim flux;

        if (SL >= 0) {
            flux = f_l;
        } else if (SL < 0 && SR > 0) {
            flux.rho = (SR * f_l.rho - SL * f_r.rho + SL * SR * (st_r.rho - st_l.rho)) / (SR - SL);
            flux.v.x = (SR * f_l.v.x - SL * f_r.v.x + SL * SR * (st_r.rho * st_r.v.x - st_l.rho * st_l.v.x)) / (SR - SL);
            flux.v.y = (SR * f_l.v.y - SL * f_r.v.y + SL * SR * (st_r.rho * st_r.v.y - st_l.rho * st_l.v.y)) / (SR - SL);
            #ifdef dim_3D
            flux.v.z = (SR * f_l.v.z - SL * f_r.v.z + SL * SR * (st_r.rho * st_r.v.z - st_l.rho * st_l.v.z)) / (SR - SL);
            #endif
            flux.E = (SR * f_l.E - SL * f_r.E + SL * SR * (st_r.E - st_l.E)) / (SR - SL);
        } else if (SR <= 0) {
            flux = f_r;
        }

        // rotate flux back to lab frame
        rotate_from_face(&flux, &g);

        return flux;
    }

    void rotate_to_face(prim* state, geom* g) {
        double velx = state->v.x;
        double vely = state->v.y;
        #ifdef dim_2D
        state->v.x = velx * g->n.x + vely * g->n.y;
        state->v.y = velx * g->m.x + vely * g->m.y;
        #else
        double velz = state->v.z;
        state->v.x = velx * g->n.x + vely * g->n.y + velz * g->n.z;
        state->v.y = velx * g->m.x + vely * g->m.y + velz * g->m.z;
        state->v.z = velx * g->p.x + vely * g->p.y + velz * g->p.z;
        #endif
    }

    void rotate_from_face(prim* state, geom* g) {
        double velx = state->v.x;
        double vely = state->v.y;
        #ifdef dim_2D
        state->v.x = velx * g->n.x + vely * g->m.x;
        state->v.y = velx * g->n.y + vely * g->m.y;
        #else
        double velz = state->v.z;
        state->v.x = velx * g->n.x + vely * g->m.x + velz * g->p.x;
        state->v.y = velx * g->n.y + vely * g->m.y + velz * g->p.y;
        state->v.z = velx * g->n.z + vely * g->m.z + velz * g->p.z;
        #endif    
    }

    geom compute_geom(double3 normal) {
        geom g;

        double nn = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
        g.n = {normal.x / nn, normal.y / nn, normal.z / nn};

        if (g.n.x != 0.0 || g.n.y != 0.0) {
            g.m = {-g.n.y, g.n.x, 0.0};
        } else {
            g.m = {1.0, 0.0, 0.0};
        }

        double mm = sqrt(g.m.x * g.m.x + g.m.y * g.m.y + g.m.z * g.m.z);
        g.m = {g.m.x / mm, g.m.y / mm, g.m.z / mm};

        g.p = {
            g.n.y * g.m.z - g.n.z * g.m.y,
            g.n.z * g.m.x - g.n.x * g.m.z,
            g.n.x * g.m.y - g.n.y * g.m.x
        };

        return g;
    }

    prim get_flux(prim* state) {

        double P = get_P_ideal_gas(state);

        // calc flux
        prim flux;

        flux.rho = state->rho * state->v.x;
        flux.v.x = state->rho * state->v.x * state->v.x + P;
        flux.v.y = state->rho * state->v.x * state->v.y;
        #ifdef dim_3D
        flux.v.z = state->rho * state->v.x * state->v.z;
        #endif
        flux.E   = (state->E + P) * state->v.x;
    
        return flux;

    }

    double get_P_ideal_gas(prim* state) {
#ifdef dim_2D
        return (_gamma_ - 1) * (state->E - (0.5 * state->rho * (state->v.x*state->v.x + state->v.y*state->v.y)));
#else   
        return (_gamma_ - 1) * (state->E - (0.5 * state->rho * (state->v.x*state->v.x + state->v.y*state->v.y + state->v.z*state->v.z)));
#endif
    }


} // namespace hydro