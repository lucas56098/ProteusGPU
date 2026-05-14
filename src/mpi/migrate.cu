#include "migrate.h"

#include "decomp.h"
#include "halo.h"
#include "global/structs.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace proteus_mpi {

// set in migrate_init from n_local_initial * ALLOC_GROWTH
static int g_n_local_max = 0;

void migrate_init(int n_local_initial) {
    g_n_local_max = max_n_local(n_local_initial);
}

#ifdef USE_MPI
// bytes-per-element wrapper for POINT_TYPE Alltoallv
static void alltoallv_point(const POINT_TYPE* send,
                            POINT_TYPE*       recv,
                            const int*        send_counts,
                            const int*        send_displs,
                            const int*        recv_counts,
                            const int*        recv_displs,
                            int               nranks,
                            MPI_Comm          comm) {
    std::vector<int> sc_b(nranks), sd_b(nranks), rc_b(nranks), rd_b(nranks);
    const int        bs = (int)sizeof(POINT_TYPE);
    for (int r = 0; r < nranks; r++) {
        sc_b[r] = send_counts[r] * bs;
        sd_b[r] = send_displs[r] * bs;
        rc_b[r] = recv_counts[r] * bs;
        rd_b[r] = recv_displs[r] * bs;
    }
    MPI_Alltoallv(send, sc_b.data(), sd_b.data(), MPI_BYTE,
                  recv, rc_b.data(), rd_b.data(), MPI_BYTE, comm);
}
#endif

void migrate_seeds(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* prim_new) {
#ifndef USE_MPI
    (void)mesh; (void)primvar; (void)prim_new;
    return;
#else
    if (g_decomp.nranks <= 1) return;

    const int    my_rank = g_decomp.rank;
    const int    nranks  = g_decomp.nranks;
    const int    N_grid  = g_decomp.N_grid_global;
    const double bf      = mesh->buff;
    const int    n_hydro = (int)mesh->n_hydro;

    POINT_TYPE* pts = mesh->scratch_move;  // new positions (in [0,1) after move_mesh fmod)

    // classify migrants per dest rank
    std::vector<int> send_counts(nranks, 0);
    std::vector<int> per_cell_dest(n_hydro, -1);
    for (int k = 0; k < n_hydro; k++) {
        const double px = pts[k].x;
        const double py = pts[k].y;
#ifdef dim_3D
        const double pz = pts[k].z;
#else
        const double pz = 0.0;
#endif
        int bx, by, bz;
        decomp_bucket_of_point(px, py, pz, N_grid, bf, &bx, &by, &bz);
        const int owner = decomp_owner_of_bucket(bx, by, bz);
        if (owner == my_rank) continue;
        if (owner < 0) {
            fprintf(stderr,
                    "[rank %d] MIGRATE: invalid owner for cell %d at (%g,%g,%g) → bucket (%d,%d,%d).\n",
                    my_rank, k, px, py, pz, bx, by, bz);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        per_cell_dest[k] = owner;
        send_counts[owner]++;
    }

    // exchange counts
    std::vector<int> recv_counts(nranks, 0);
    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 g_decomp.cart_comm);

    int total_send = 0, total_recv = 0;
    for (int r = 0; r < nranks; r++) {
        total_send += send_counts[r];
        total_recv += recv_counts[r];
    }

    // global short-circuit if no rank has any migrants
    int local_any  = (total_send + total_recv > 0) ? 1 : 0;
    int global_any = 0;
    MPI_Allreduce(&local_any, &global_any, 1, MPI_INT, MPI_LOR, g_decomp.cart_comm);
    if (!global_any) return;

    // displacements + pack send buffers
    std::vector<int> send_displs(nranks, 0);
    std::vector<int> recv_displs(nranks, 0);
    for (int r = 1; r < nranks; r++) send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
    for (int r = 1; r < nranks; r++) recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];

    std::vector<POINT_TYPE> sb_pos(total_send);
    std::vector<double>     sb_rho_old(total_send), sb_E_old(total_send);
    std::vector<POINT_TYPE> sb_v_old(total_send);
    std::vector<double>     sb_rho_new(total_send), sb_E_new(total_send);
    std::vector<POINT_TYPE> sb_v_new(total_send);
#ifdef MOVING_MESH
    std::vector<POINT_TYPE> sb_vmesh(total_send);
    std::vector<double>     sb_oldvol(total_send);
#endif

    std::vector<int> migrant_local_k;
    migrant_local_k.reserve(total_send);
    {
        std::vector<int> cursor = send_displs;
        for (int k = 0; k < n_hydro; k++) {
            const int dest = per_cell_dest[k];
            if (dest < 0) continue;
            const int slot   = cursor[dest]++;
            sb_pos[slot]     = pts[k];
            sb_rho_old[slot] = primvar->rho[k];
            sb_v_old[slot]   = primvar->v[k];
            sb_E_old[slot]   = primvar->E[k];
            sb_rho_new[slot] = prim_new->rho[k];
            sb_v_new[slot]   = prim_new->v[k];
            sb_E_new[slot]   = prim_new->E[k];
#ifdef MOVING_MESH
            sb_vmesh[slot]   = mesh->v_mesh[k];
            sb_oldvol[slot]  = mesh->old_volumes[k];
#endif
            migrant_local_k.push_back(k);
        }
    }

    // exchange
    std::vector<POINT_TYPE> rb_pos(total_recv);
    std::vector<double>     rb_rho_old(total_recv), rb_E_old(total_recv);
    std::vector<POINT_TYPE> rb_v_old(total_recv);
    std::vector<double>     rb_rho_new(total_recv), rb_E_new(total_recv);
    std::vector<POINT_TYPE> rb_v_new(total_recv);
#ifdef MOVING_MESH
    std::vector<POINT_TYPE> rb_vmesh(total_recv);
    std::vector<double>     rb_oldvol(total_recv);
#endif

    auto a2av_d = [&](const double* s, double* r) {
        MPI_Alltoallv(s, send_counts.data(), send_displs.data(), MPI_DOUBLE,
                      r, recv_counts.data(), recv_displs.data(), MPI_DOUBLE,
                      g_decomp.cart_comm);
    };
    auto a2av_p = [&](const POINT_TYPE* s, POINT_TYPE* r) {
        alltoallv_point(s, r, send_counts.data(), send_displs.data(),
                        recv_counts.data(), recv_displs.data(), nranks, g_decomp.cart_comm);
    };

    a2av_p(sb_pos.data(), rb_pos.data());
    a2av_d(sb_rho_old.data(), rb_rho_old.data());
    a2av_p(sb_v_old.data(), rb_v_old.data());
    a2av_d(sb_E_old.data(), rb_E_old.data());
    a2av_d(sb_rho_new.data(), rb_rho_new.data());
    a2av_p(sb_v_new.data(), rb_v_new.data());
    a2av_d(sb_E_new.data(), rb_E_new.data());
#ifdef MOVING_MESH
    a2av_p(sb_vmesh.data(), rb_vmesh.data());
    a2av_d(sb_oldvol.data(), rb_oldvol.data());
#endif

    // compact local arrays via swap-with-last-and-shrink (descending so pops don't disturb earlier indices)
    std::sort(migrant_local_k.begin(), migrant_local_k.end(), std::greater<int>());
    int n_after_remove = n_hydro;
    for (int k_remove : migrant_local_k) {
        const int k_last = n_after_remove - 1;
        if (k_remove != k_last) {
            pts[k_remove]                    = pts[k_last];
            primvar->rho[k_remove]           = primvar->rho[k_last];
            primvar->v[k_remove]             = primvar->v[k_last];
            primvar->E[k_remove]             = primvar->E[k_last];
            prim_new->rho[k_remove]          = prim_new->rho[k_last];
            prim_new->v[k_remove]            = prim_new->v[k_last];
            prim_new->E[k_remove]            = prim_new->E[k_last];
#ifdef MOVING_MESH
            mesh->v_mesh[k_remove]           = mesh->v_mesh[k_last];
            mesh->old_volumes[k_remove]      = mesh->old_volumes[k_last];
#endif
            mesh->cell_to_original[k_remove] = mesh->cell_to_original[k_last];
        }
        n_after_remove--;
    }

    // capacity bound
    const int n_new = n_after_remove + total_recv;
    if (n_new > g_n_local_max) {
        fprintf(stderr,
                "[rank %d] MIGRATE: n_hydro_new=%d > n_local_max=%d. "
                "Increase ALLOC_GROWTH in src/mpi/extension.h or rebalance the IC.\n",
                my_rank, n_new, g_n_local_max);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // append received cells
    for (int j = 0; j < total_recv; j++) {
        const int k                = n_after_remove + j;
        pts[k]                     = rb_pos[j];
#ifdef dim_3D
        mesh->seeds[k]             = double3{rb_pos[j].x, rb_pos[j].y, rb_pos[j].z};
#else
        mesh->seeds[k]             = double3{rb_pos[j].x, rb_pos[j].y, 0.0};
#endif
        primvar->rho[k]            = rb_rho_old[j];
        primvar->v[k]              = rb_v_old[j];
        primvar->E[k]              = rb_E_old[j];
        prim_new->rho[k]           = rb_rho_new[j];
        prim_new->v[k]             = rb_v_new[j];
        prim_new->E[k]             = rb_E_new[j];
#ifdef MOVING_MESH
        mesh->v_mesh[k]            = rb_vmesh[j];
        mesh->old_volumes[k]       = rb_oldvol[j];
#endif
        mesh->cell_to_original[k]  = (unsigned int)k;
    }

    mesh->n_hydro = (hsize_t)n_new;

    // cell-count conservation check
    int n_global = 0;
    MPI_Allreduce(&n_new, &n_global, 1, MPI_INT, MPI_SUM, g_decomp.cart_comm);
    static int s_n_total_expected = 0;
    if (s_n_total_expected == 0) s_n_total_expected = n_global;
    if (n_global != s_n_total_expected) {
        if (my_rank == 0) {
            fprintf(stderr,
                    "MIGRATE: FATAL global cell-count drift: %d != %d. A cell was duplicated or lost.\n",
                    n_global, s_n_total_expected);
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#endif
}

}  // namespace proteus_mpi
