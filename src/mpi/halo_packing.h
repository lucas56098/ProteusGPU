#ifndef MPI_HALO_PACKING_H
#define MPI_HALO_PACKING_H
#pragma once

// Bodies of the halo pack and unpack loops, the same code on CPU and GPU.

#include "global/gpu_compat.h"
#include "global/math_utils.h"
#include "global/structs.h"
#include "halo.h"

namespace proteus_mpi {
    namespace pack {

        HD inline void
        pack_prim_body(int s, const int* used_export_indices, const hydro::primvars* primvar, HaloPrimCell* sendbuf) {
            const int    k = used_export_indices[s];
            HaloPrimCell pkt;
            pkt.rho    = primvar->rho[k];
            pkt.v      = primvar->v[k];
            pkt.E      = primvar->E[k];
            sendbuf[s] = pkt;
        }

        HD inline void
        unpack_prim_body(int s, const int* used_to_full_slot, const HaloPrimCell* recvbuf, hydro::primvars* primvar) {
            const HaloPrimCell pkt  = recvbuf[s];
            const int          slot = used_to_full_slot[s];
            primvar->rho_g[slot]    = pkt.rho;
            primvar->v_g[slot]      = pkt.v;
            primvar->E_g[slot]      = pkt.E;
        }

        HD inline void pack_grad_body(int                             slot,
                                      const int*                      used_export_indices,
                                      const gradients::PrimGradients* grads,
                                      POINT_TYPE*                     sendbuf) {
            const int k      = used_export_indices[slot];
            const int s      = slot * HALO_GRAD_COMPONENTS;
            int       c      = 0;
            sendbuf[s + c++] = grads->rho[k];
            sendbuf[s + c++] = grads->vx[k];
            sendbuf[s + c++] = grads->vy[k];
#ifdef dim_3D
            sendbuf[s + c++] = grads->vz[k];
#endif
            sendbuf[s + c++] = grads->E[k];
            sendbuf[s + c++] = grads->anchor[k];
        }

        HD inline void unpack_grad_body(int                       slot,
                                        const int*                used_to_full_slot,
                                        const POINT_TYPE*         recvbuf,
                                        gradients::PrimGradients* grads) {
            const int g     = used_to_full_slot[slot];
            const int s     = slot * HALO_GRAD_COMPONENTS;
            int       c     = 0;
            grads->rho_g[g] = recvbuf[s + c++];
            grads->vx_g[g]  = recvbuf[s + c++];
            grads->vy_g[g]  = recvbuf[s + c++];
#ifdef dim_3D
            grads->vz_g[g] = recvbuf[s + c++];
#endif
            grads->E_g[g]      = recvbuf[s + c++];
            grads->anchor_g[g] = recvbuf[s + c++];
        }

        // centroid - seed of an exported cell
        HD inline void pack_com_off_body(
            int s, const int* used_export_indices, const double3* com, const double3* seeds, POINT_TYPE* sendbuf) {
            const int k = used_export_indices[s];
            sendbuf[s]  = point_diff_periodic(com[k], seeds[k]);
        }

        HD inline void
        unpack_com_off_body(int slot, const int* used_to_full_slot, const POINT_TYPE* recvbuf, POINT_TYPE* com_off_g) {
            const int g  = used_to_full_slot[slot];
            com_off_g[g] = recvbuf[slot];
        }

#ifdef MOVING_MESH
        HD inline void
        pack_v_mesh_body(int s, const int* used_export_indices, const POINT_TYPE* v_mesh, POINT_TYPE* sendbuf) {
            const int k = used_export_indices[s];
            sendbuf[s]  = v_mesh[k];
        }

        HD inline void
        unpack_v_mesh_body(int slot, const int* used_to_full_slot, const POINT_TYPE* recvbuf, POINT_TYPE* v_mesh_g) {
            const int g = used_to_full_slot[slot];
            v_mesh_g[g] = recvbuf[slot];
        }
#endif

#ifdef VOL_REGULARIZE
        HD inline void pack_vol_body(int s, const int* used_export_indices, const double* volumes, double* sendbuf) {
            const int k = used_export_indices[s];
            sendbuf[s]  = volumes[k];
        }

        HD inline void
        unpack_vol_body(int slot, const int* used_to_full_slot, const double* recvbuf, double* volumes_g) {
            const int g  = used_to_full_slot[slot];
            volumes_g[g] = recvbuf[slot];
        }
#endif

        // a face to an MPI ghost marks that ghost as used
        HD inline void mark_used_bitmap_body(
            int f, const int* neighbor_cell, int mpi_base, int mpi_top, unsigned char* recv_used_bitmap) {
            const int kn = neighbor_cell[f];
            if (kn < mpi_base || kn >= mpi_top) return;
            recv_used_bitmap[kn - mpi_base] = 1;
        }

    } // namespace pack
} // namespace proteus_mpi

#endif
