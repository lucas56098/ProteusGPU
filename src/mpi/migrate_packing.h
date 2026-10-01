#ifndef MPI_MIGRATE_PACKING_H
#define MPI_MIGRATE_PACKING_H
#pragma once

// Bodies of the migration loops, the same code on CPU and GPU.

#include "global/gpu_compat.h"
#include "global/structs.h"

namespace proteus_mpi {
    namespace pack {

        template <typename MigrantCell>
        // one leaving cell into its place in the send buffer
        HD inline void pack_migrant_body(int               k,
                                         const int*        per_cell_slot,
                                         const int*        dest_pos,
                                         const POINT_TYPE* pts,
                                         const double*     primvar_rho,
                                         const POINT_TYPE* primvar_v,
                                         const double*     primvar_E,
                                         const double*     mass,
                                         const POINT_TYPE* momentum,
                                         const double*     energy,
#ifdef MOVING_MESH
                                         const POINT_TYPE*               v_mesh,
                                         const gradients::PrimGradients* grads,
#endif
                                         MigrantCell* sendbuf) {
            if (per_cell_slot[k] < 0) return;
            MigrantCell mc;
            mc.pos      = pts[k];
            mc.rho_old  = primvar_rho[k];
            mc.v_old    = primvar_v[k];
            mc.E_old    = primvar_E[k];
            mc.mass     = mass[k];
            mc.momentum = momentum[k];
            mc.energy   = energy[k];
#ifdef MOVING_MESH
            mc.v_mesh = v_mesh[k];
            mc.grad   = grads->load(k);
#endif
            sendbuf[dest_pos[k]] = mc;
        }

        template <typename MigrantCell>
        HD inline void unpack_migrant_body(int                j,
                                           int                n_after_remove,
                                           const MigrantCell* recvbuf,
#ifdef MOVING_MESH
                                           POINT_TYPE*               v_mesh,
                                           gradients::PrimGradients* grads,
#endif
                                           POINT_TYPE* pts,
                                           double3*    seeds,
                                           double*     primvar_rho,
                                           POINT_TYPE* primvar_v,
                                           double*     primvar_E,
                                           double*     mass,
                                           POINT_TYPE* momentum,
                                           double*     energy) {
            const int         k  = n_after_remove + j;
            const MigrantCell mc = recvbuf[j];
            pts[k]               = mc.pos;
#ifdef dim_3D
            seeds[k] = double3{mc.pos.x, mc.pos.y, mc.pos.z};
#else
            seeds[k] = double3{mc.pos.x, mc.pos.y, 0.0};
#endif
            primvar_rho[k] = mc.rho_old;
            primvar_v[k]   = mc.v_old;
            primvar_E[k]   = mc.E_old;
            mass[k]        = mc.mass;
            momentum[k]    = mc.momentum;
            energy[k]      = mc.energy;
#ifdef MOVING_MESH
            v_mesh[k]     = mc.v_mesh;
            grads->rho[k] = mc.grad.rho;
            grads->vx[k]  = mc.grad.vx;
            grads->vy[k]  = mc.grad.vy;
#ifdef dim_3D
            grads->vz[k] = mc.grad.vz;
#endif
            grads->E[k]      = mc.grad.E;
            grads->anchor[k] = mc.grad.anchor;
#endif
        }

    } // namespace pack
} // namespace proteus_mpi

#endif
