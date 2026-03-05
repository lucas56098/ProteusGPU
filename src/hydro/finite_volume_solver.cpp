#include "finite_volume_solver.h"
#include "../global/allvars.h"
#include "riemann.h"

namespace hydro {

    // init hydrostruct from IC data
    primvars* init(int n_hydro) {

        // allocate prim struct
        primvars* hydro_data = new primvars();
        hydro_data->rho = (double*)malloc(n_hydro * sizeof(double));
        hydro_data->v = (POINT_TYPE*)malloc(n_hydro * sizeof(POINT_TYPE));
        hydro_data->E = (double*)malloc(n_hydro * sizeof(double));

        // fill hydro_data from icData
        for (int i = 0; i < n_hydro; i++) {
            hydro_data->rho[i] = icData.rho[i];
            hydro_data->E[i] = icData.Energy[i];
            
            hydro_data->v[i].x = icData.vel[DIMENSION * i];
            hydro_data->v[i].y = icData.vel[DIMENSION * i + 1];
            #ifdef dim_3D
            hydro_data->v[i].z = icData.vel[DIMENSION * i + 2];
            #endif
        }

        std::cout << "HYDRO: Initialized primitive variables for " << n_hydro << " particles" << std::endl;

        return hydro_data;
    }

    // free the primvars again
    void free_prim(primvars** primvar) {
        free((*primvar)->rho);
        free((*primvar)->v);
        free((*primvar)->E);
        free(*primvar);
        *primvar = NULL;
    }

    // main hydro routine (computes fluxes and updates states)
    void hydro_step(double dt, const VMesh* mesh, primvars* primvar) {

        // new primvars
        primvars new_prim;
        new_prim.rho = (double*)malloc(mesh->n_hydro*sizeof(double));
        new_prim.v = (POINT_TYPE*)malloc(mesh->n_hydro*sizeof(POINT_TYPE));
        new_prim.E = (double*)malloc(mesh->n_hydro*sizeof(double));


        // loop over all active cells to calc new primvars
        #pragma omp parallel for
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {

            // get state of cell i
            prim state_i;
            state_i.rho = primvar->rho[i];
            state_i.E = primvar->E[i];
            state_i.v.x = primvar->v[i].x;
            state_i.v.y = primvar->v[i].y;
            #ifdef dim_3D
            state_i.v.z = primvar->v[i].z;
            #endif

            prim total_flux;

            // calculate total_flux by summing over edge flux * edge_length
            for (hsize_t j = 0; j < mesh->face_counts[i]; j++) {

                // get value of other cell
                prim state_j = get_state_j(i, j, mesh, primvar);

                // calc flux using riemann solver
                prim flux_ij = riemann_hll(i, j, state_i, state_j, mesh);

                // get face area/length
                double face_area = mesh->face_area[mesh->face_ptr[i] + j];

                // add to total flux * area
                total_flux.rho += flux_ij.rho * face_area;
                total_flux.v.x += flux_ij.v.x * face_area;
                total_flux.v.y += flux_ij.v.y * face_area;
                #ifdef dim_3D
                total_flux.v.z += flux_ij.v.z * face_area;
                #endif
                total_flux.E += flux_ij.E * face_area;
                
            }

            double V = mesh->volumes[i];
            double frac = dt/V;


            new_prim.rho[i] = state_i.rho - frac * total_flux.rho;

            double new_rho_inv = 1.0 / new_prim.rho[i];

            new_prim.v[i].x = (state_i.rho * state_i.v.x - frac * total_flux.v.x) * new_rho_inv;
            new_prim.v[i].y = (state_i.rho * state_i.v.y - frac * total_flux.v.y) * new_rho_inv;
            #ifdef dim_3D
            new_prim.v[i].z = (state_i.rho * state_i.v.z - frac * total_flux.v.z) * new_rho_inv;
            #endif

            new_prim.E[i] = state_i.E - frac * total_flux.E;
        }

        #pragma omp parallel for
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            primvar->rho[i] = new_prim.rho[i];
            primvar->v[i].x = new_prim.v[i].x;
            primvar->v[i].y = new_prim.v[i].y;
            #ifdef dim_3D
            primvar->v[i].z = new_prim.v[i].z;
            #endif
            primvar->E[i] = new_prim.E[i];
        }

        // free new primvars
        free(new_prim.rho);
        free(new_prim.v);
        free(new_prim.E);
    }

    prim get_state_j(hsize_t i, int j, const VMesh* mesh, primvars* primvar) {

        prim state_j;

        // get index
        hsize_t index = mesh->neighbor_cell[mesh->face_ptr[i] + j];
        hsize_t n_hydro = (hsize_t)mesh->n_hydro;

        if (index >= n_hydro) {
            // its a ghost cell (so lets get the correct index)
            index = mesh->ghost_ids[index - n_hydro];
        }

        // load the state
        state_j.rho = primvar->rho[index];
        state_j.v.x = primvar->v[index].x;
        state_j.v.y = primvar->v[index].y;
        #ifdef dim_3D
        state_j.v.z = primvar->v[index].z;
        #endif
        state_j.E = primvar->E[index];

        return state_j;
    }

    // calc timestep using CFL condition for euler equations
    double dt_CFL(double CFL, const VMesh* mesh, const primvars* primvar) {

        double min_dt = 1e100;

        #pragma omp parallel for reduction(min : min_dt)
        for (hsize_t i = 0; i < mesh->n_hydro; i++) {
            // build prim state for cell i to get pressure
            prim state_i;
            state_i.rho = primvar->rho[i];
            state_i.E = primvar->E[i];
            state_i.v.x = primvar->v[i].x;
            state_i.v.y = primvar->v[i].y;
            #ifdef dim_3D
            state_i.v.z = primvar->v[i].z;
            #endif

            double P = get_P_ideal_gas(&state_i);
            double c_i = sqrt(gamma * P / state_i.rho);

            #ifdef dim_2D
            double R_i = sqrt(mesh->volumes[i] / M_PI);
            double v_abs = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y);
            #else
            double R_i = cbrt(3.0 * mesh->volumes[i] / (4.0 * M_PI));
            double v_abs = sqrt(state_i.v.x * state_i.v.x + state_i.v.y * state_i.v.y + state_i.v.z * state_i.v.z);
            #endif

            double dt_i = CFL * (R_i / (c_i + v_abs));

            if (dt_i < min_dt) {
                min_dt = dt_i;
            }
        }

        return min_dt;
    }

} // namespace hydro