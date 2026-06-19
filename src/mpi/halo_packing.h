#ifndef MPI_HALO_PACKING_H
#define MPI_HALO_PACKING_H
#pragma once

// Per-element halo pack/unpack bodies. Same function called from both an
// `__global__` CUDA kernel (CUDA build) and an OpenMP loop (CPU_DEBUG build);
// behaviour is identical by construction. Bodies are pure: no globals, no MPI,
// no allocation — every input pointer is passed explicitly.

#include "global/gpu_compat.h"
#include "global/structs.h"
#include "halo.h"

namespace proteus_mpi {
    namespace pack {

        // ============================================================
        // Seeds (full halo: every export slot ships)
        // ============================================================

        // neighbor_shift_flat is a flat HALO_MAX_NEIGHBORS*3 array (host's [n][a] flattened
        // row-major). The kernel can't take a 2D-array argument cleanly across CUDA/HD,
        // so we flatten the indexing here.
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

        // is_outer_layer fill: one thread per direction n, marks the first
        // recv_n_outer[n] slots in that direction's receive range as outer (1),
        // remaining as inner (0). Bodies are *not* per-element here — caller
        // launches with one work item per neighbor direction.
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

        // ============================================================
        // Primvars (used subset)
        // ============================================================

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

        // ============================================================
        // Gradients (used subset, N_COMP = 3 + DIMENSION fields per cell)
        // ============================================================

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

        // ============================================================
        // Mesh velocity (used subset, MOVING_MESH only)
        // ============================================================

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

        // ============================================================
        // Used-recv bitmap construction (idempotent writes of 1 — race-safe)
        // ============================================================

        HD inline void mark_used_bitmap_body(
            int f, const int* neighbor_cell, int mpi_base, int mpi_top, unsigned char* recv_used_bitmap) {
            const int kn = neighbor_cell[f];
            if (kn < mpi_base || kn >= mpi_top) return;
            recv_used_bitmap[kn - mpi_base] = 1;
        }

    } // namespace pack
} // namespace proteus_mpi

#endif // MPI_HALO_PACKING_H
