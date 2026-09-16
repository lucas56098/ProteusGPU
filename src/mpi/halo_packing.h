#ifndef MPI_HALO_PACKING_H
#define MPI_HALO_PACKING_H
#pragma once

// Bodies of the halo pack and unpack loops, the same code on CPU and GPU.

#include "global/gpu_compat.h"
#include "global/structs.h"
#include "halo.h"

namespace proteus_mpi {
    namespace pack {

        // an exported seed, shifted if its neighbour is across the box border
        HD inline void pack_seed_body(int                  s,
                                      const POINT_TYPE*    pts,
                                      const int*           export_indices,
                                      const unsigned char* dir_of_slot,
                                      const double*        neighbor_shift_flat,
                                      POINT_TYPE*          sendbuf) {
            const int  n = (int)dir_of_slot[s];
            const int  k = export_indices[s];
            POINT_TYPE p = pts[k];
            p.x += neighbor_shift_flat[n * 3 + 0];
            p.y += neighbor_shift_flat[n * 3 + 1];
#ifdef dim_3D
            p.z += neighbor_shift_flat[n * 3 + 2];
#endif
            sendbuf[s] = p;
        }

        // a received seed becomes a ghost: one point for the build, one seed for the mesh
        HD inline void
        unpack_seed_body(int slot, int pts_mpi_base, const POINT_TYPE* recvbuf, POINT_TYPE* pts, double3* seeds_g) {
            const POINT_TYPE p     = recvbuf[slot];
            const int        pts_k = pts_mpi_base + slot;
            pts[pts_k]             = p;
#ifdef dim_3D
            seeds_g[slot] = double3{p.x, p.y, p.z};
#else
            seeds_g[slot] = double3{p.x, p.y, 0.0};
#endif
        }

        // the ghosts from the outermost bucket layer come first
        HD inline void fill_is_outer_layer_body(int            n,
                                                const int*     recv_n_outer,
                                                const int*     ghost_offset,
                                                const int*     recv_count,
                                                unsigned char* is_outer_layer) {
            const int base  = ghost_offset[n];
            const int n_out = recv_n_outer[n];
            const int n_tot = recv_count[n];
            for (int j = 0; j < n_out; j++)
                is_outer_layer[base + j] = 1;
            for (int j = n_out; j < n_tot; j++)
                is_outer_layer[base + j] = 0;
        }

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
            const int N_COMP = 3 + DIMENSION;
            const int k      = used_export_indices[slot];
            const int s      = slot * N_COMP;
            int       c      = 0;
            sendbuf[s + c++] = grads->rho[k];
            sendbuf[s + c++] = grads->vx[k];
            sendbuf[s + c++] = grads->vy[k];
#ifdef dim_3D
            sendbuf[s + c++] = grads->vz[k];
#endif
            sendbuf[s + c++] = grads->E[k];
        }

        HD inline void unpack_grad_body(int                       slot,
                                        const int*                used_to_full_slot,
                                        const POINT_TYPE*         recvbuf,
                                        gradients::PrimGradients* grads) {
            const int N_COMP = 3 + DIMENSION;
            const int g      = used_to_full_slot[slot];
            const int s      = slot * N_COMP;
            int       c      = 0;
            grads->rho_g[g]  = recvbuf[s + c++];
            grads->vx_g[g]   = recvbuf[s + c++];
            grads->vy_g[g]   = recvbuf[s + c++];
#ifdef dim_3D
            grads->vz_g[g] = recvbuf[s + c++];
#endif
            grads->E_g[g] = recvbuf[s + c++];
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
