#ifndef MPI_MIGRATE_PACKING_H
#define MPI_MIGRATE_PACKING_H
#pragma once

// Per-element migrate pack/unpack bodies. Same HD-inline functions called from
// both __global__ CUDA kernels (CUDA build) and OpenMP loops (CPU_DEBUG build).
// All bodies are pure: no globals, no MPI calls. The MigrantCell type is the
// caller's (defined in migrate.cu) and forwarded as a template-free POD type.

#include "decomp.h"
#include "global/gpu_compat.h"
#include "global/structs.h"

namespace proteus_mpi {
    namespace pack {

        // ============================================================
        // assign_destinations: per-cell bucket -> owner-rank lookup, atomic count
        // ============================================================

        // for a single cell k, compute its post-advance bucket, find the owner, decide
        // whether to migrate (= owner != my_rank). On a migrant, returns:
        //   slot_id_out >= 0  (per-step variant: neighbor-slot index from neighbor_lookup,
        //                      rebalance variant: dest_rank directly)
        //   slot_id_out == -1 -> non-migrant (stays local)
        //   slot_id_out == -2 -> invalid owner (caller exits)
        //   slot_id_out == -3 -> not a Cart neighbor (caller exits; CFL violation)
        //
        // For per-step (variant=0) caller supplies a neighbor_rank->slot lookup via
        // a flat array (rank -> slot, or -1 if not a neighbor). For rebalance
        // (variant=1) the slot IS the rank.
        template <typename M_PerCellSlot, typename M_SendCounts>
        HD inline void assign_destination_body(int               k,
                                               const POINT_TYPE* pts,
                                               int               my_rank,
                                               int               N_grid_global,
                                               double            buff,
                                               int               dims_x,
                                               int               dims_y,
                                               int               dims_z,
                                               const int*        splits_x,
                                               const int*        splits_y,
                                               const int*        splits_z,
                                               const int*        coord_to_rank,
                                               const int*     neighbor_rank_to_slot, // size nranks; -1 for non-neighbor
                                               int            variant,
                                               M_PerCellSlot* per_cell_slot,
                                               M_SendCounts*  send_counts,
                                               int*           error_flag) {
            const double px = pts[k].x;
            const double py = pts[k].y;
#ifdef dim_3D
            const double pz = pts[k].z;
#else
            const double pz = 0.0;
#endif
            int bx, by, bz;
            decomp_bucket_of_point(px, py, pz, N_grid_global, buff, &bx, &by, &bz);
            const int owner = decomp_owner_of_bucket_dev(
                bx, by, bz, N_grid_global, dims_x, dims_y, dims_z, splits_x, splits_y, splits_z, coord_to_rank);
            if (owner == my_rank) {
                per_cell_slot[k] = -1;
                return;
            }
            if (owner < 0) {
                per_cell_slot[k] = -1;
                portable_atomicExch(error_flag, 1); // ERR_INVALID_OWNER
                return;
            }
            int slot;
            if (variant == 1) {
                // rebalance: dest = rank itself
                slot = owner;
            } else {
                // per-step: owner must be one of our Cart neighbors
                slot = neighbor_rank_to_slot[owner];
                if (slot < 0) {
                    per_cell_slot[k] = -1;
                    portable_atomicExch(error_flag, 2); // ERR_NOT_NEIGHBOR
                    return;
                }
            }
            per_cell_slot[k] = slot;
            portable_atomicAdd(&send_counts[slot], 1);
        }

        // ============================================================
        // pack_outgoing_migrants: per-cell scatter into sendbuf via atomic cursor
        // ============================================================

        template <typename MigrantCell>
        HD inline void pack_migrant_body(int               k,
                                         const int*        per_cell_slot,
                                         const POINT_TYPE* pts,
                                         const double*     primvar_rho,
                                         const POINT_TYPE* primvar_v,
                                         const double*     primvar_E,
                                         const double*     prim_new_rho,
                                         const POINT_TYPE* prim_new_v,
                                         const double*     prim_new_E,
#ifdef MOVING_MESH
                                         const POINT_TYPE* v_mesh,
                                         const double*     old_volumes,
#endif
                                         int*         cursor,
                                         MigrantCell* sendbuf,
                                         int*         n_migrant_local_counter,
                                         int*         migrant_local_k) {
            const int slot_id = per_cell_slot[k];
            if (slot_id < 0) return;
            const int   slot = portable_atomicAdd(&cursor[slot_id], 1);
            MigrantCell mc;
            mc.pos     = pts[k];
            mc.rho_old = primvar_rho[k];
            mc.v_old   = primvar_v[k];
            mc.E_old   = primvar_E[k];
            mc.rho_new = prim_new_rho[k];
            mc.v_new   = prim_new_v[k];
            mc.E_new   = prim_new_E[k];
#ifdef MOVING_MESH
            mc.v_mesh     = v_mesh[k];
            mc.old_volume = old_volumes[k];
#endif
            sendbuf[slot]              = mc;
            const int local_idx        = portable_atomicAdd(n_migrant_local_counter, 1);
            migrant_local_k[local_idx] = k;
        }

        // ============================================================
        // append_incoming_migrants: per-recv scatter into local arrays
        // ============================================================

        template <typename MigrantCell>
        HD inline void unpack_migrant_body(int                j,
                                           int                n_after_remove,
                                           const MigrantCell* recvbuf,
                                           POINT_TYPE*        pts,
                                           double3*           seeds,
                                           double*            primvar_rho,
                                           POINT_TYPE*        primvar_v,
                                           double*            primvar_E,
                                           double*            prim_new_rho,
                                           POINT_TYPE*        prim_new_v,
                                           double*            prim_new_E,
#ifdef MOVING_MESH
                                           POINT_TYPE* v_mesh,
                                           double*     old_volumes,
#endif
                                           unsigned int* cell_to_original) {
            const int         k  = n_after_remove + j;
            const MigrantCell mc = recvbuf[j];
            pts[k]               = mc.pos;
#ifdef dim_3D
            seeds[k] = double3{mc.pos.x, mc.pos.y, mc.pos.z};
#else
            seeds[k] = double3{mc.pos.x, mc.pos.y, 0.0};
#endif
            primvar_rho[k]  = mc.rho_old;
            primvar_v[k]    = mc.v_old;
            primvar_E[k]    = mc.E_old;
            prim_new_rho[k] = mc.rho_new;
            prim_new_v[k]   = mc.v_new;
            prim_new_E[k]   = mc.E_new;
#ifdef MOVING_MESH
            v_mesh[k]      = mc.v_mesh;
            old_volumes[k] = mc.old_volume;
#endif
            cell_to_original[k] = (unsigned int)k;
        }

    } // namespace pack
} // namespace proteus_mpi

#endif // MPI_MIGRATE_PACKING_H
