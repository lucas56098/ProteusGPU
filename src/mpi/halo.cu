#include "halo.h"

#include "decomp.h"
#include "global/structs.h"
#include "gradients/gradients.h"
#include "voronoi/voronoi.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace proteus_mpi {

MpiHalo g_halo            = {};
int     g_n_mpi_capacity  = 0;  // mirrors g_halo.n_mpi_capacity for the include-cycle-free extension.h

// ============================================================
// Init / free
// ============================================================

// build the Cart-neighbor table; skip directions that fold back to ourselves
// (those are handled by the existing local periodic-ghost layer)
static void build_neighbor_table() {
    g_halo.n_neighbors = 0;

#ifdef USE_MPI
    int my_coords[3];
    MPI_Cart_coords(g_decomp.cart_comm, g_decomp.rank, 3, my_coords);

#ifdef dim_3D
    const int dz_lo = -1, dz_hi = 1;
#else
    const int dz_lo = 0, dz_hi = 0;
#endif

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = dz_lo; dz <= dz_hi; dz++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;

                int coords[3]     = {my_coords[0] + dx, my_coords[1] + dy, my_coords[2] + dz};
                int neighbor_rank = 0;
                MPI_Cart_rank(g_decomp.cart_comm, coords, &neighbor_rank);

                // skip directions that wrap back to self via a length-1 axis (handled
                // by local periodic ghosts). Mixed wraps (length-1 axis + multi-rank
                // shift) still reach a different rank and need MPI exchange, otherwise
                // corner cells miss their diagonal neighbors.
                if (neighbor_rank == g_decomp.rank) continue;

                // periodic-wrap shift: sender's seeds need ±1 along any axis where
                // (my_coords + d) lies outside [0, dims). Mirrors the local periodic
                // ghost sx ∈ {-1,0,+1} shift in periodic_mesh.cu.
                double shift[3] = {0.0, 0.0, 0.0};
                for (int a = 0; a < 3; a++) {
                    if (coords[a] < 0) shift[a] = +1.0;
                    if (coords[a] >= g_decomp.dims[a]) shift[a] = -1.0;
                }

                int n                       = g_halo.n_neighbors;
                g_halo.neighbor_ranks[n]    = neighbor_rank;
                g_halo.neighbor_dirs[n][0]  = dx;
                g_halo.neighbor_dirs[n][1]  = dy;
                g_halo.neighbor_dirs[n][2]  = dz;
                g_halo.neighbor_shift[n][0] = shift[0];
                g_halo.neighbor_shift[n][1] = shift[1];
                g_halo.neighbor_shift[n][2] = shift[2];
                g_halo.n_neighbors++;
            }
        }
    }
#endif
}

void halo_init(int n_local, double buff) {
#ifndef USE_MPI
    (void)n_local;
    (void)buff;
    g_halo.n_neighbors      = 0;
    g_halo.n_mpi_capacity   = 0;
    g_halo.per_dir_capacity = 0;
    return;
#else
    g_halo.comm = g_decomp.cart_comm;
    build_neighbor_table();

    if (g_halo.n_neighbors == 0) {
        // single-rank Cart or all neighbors fold back to self — no halo needed;
        // cart_comm is still valid for the dt-Allreduce when nranks > 1.
        g_halo.n_mpi_capacity   = 0;
        g_halo.per_dir_capacity = 0;
        g_n_mpi_capacity        = 0;
        if (g_decomp.rank == 0) {
            printf("HALO: 0 Cart neighbors (single-rank topology) — halo disabled.\n");
            fflush(stdout);
        }
        return;
    }

    // per-direction capacity, uniform-density assumption with face-bound margin.
    // For halo layer of W buckets, the face term ~ n_local * W / B_axis dominates
    // when W < min(B); corners pay O(W^3 / (Bx*By*Bz)) which is smaller in that regime.
    // SAFETY absorbs non-uniformity and the iterative-widening overshoot.
    constexpr int    MAX_WIDEN_ITERS = 4;
    constexpr double SAFETY          = 2.0;
    const int        W_alloc         = halo_default_width(buff) * 2 + 2 * (MAX_WIDEN_ITERS - 1);

    int min_B = g_decomp.b1[0] - g_decomp.b0[0];
    for (int a = 1; a < DIMENSION; a++) {
        const int Ba = g_decomp.b1[a] - g_decomp.b0[a];
        if (Ba < min_B) min_B = Ba;
    }
    if (min_B < 1) min_B = 1;

    const long long est       = (long long)std::ceil(SAFETY * (double)n_local * (double)W_alloc / (double)min_B);
    const int        floor_cap = 64;  // small-problem floor
    g_halo.per_dir_capacity    = (int)std::max<long long>(floor_cap, est);
    g_halo.n_mpi_capacity      = HALO_MAX_NEIGHBORS * g_halo.per_dir_capacity;
    g_n_mpi_capacity           = g_halo.n_mpi_capacity;

    const int total_cap = g_halo.n_mpi_capacity;

    g_halo.export_indices = (int*)gpu_malloc(sizeof(int) * total_cap);

    g_halo.sendbuf_seed  = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap);
    g_halo.recvbuf_seed  = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap);
    g_halo.sendbuf_rho   = (double*)gpu_malloc(sizeof(double) * total_cap);
    g_halo.recvbuf_rho   = (double*)gpu_malloc(sizeof(double) * total_cap);
    g_halo.sendbuf_v     = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap);
    g_halo.recvbuf_v     = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap);
    g_halo.sendbuf_E     = (double*)gpu_malloc(sizeof(double) * total_cap);
    g_halo.recvbuf_E     = (double*)gpu_malloc(sizeof(double) * total_cap);
    g_halo.sendbuf_vmesh = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap);
    g_halo.recvbuf_vmesh = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap);

    const int grad_components = 3 + DIMENSION;  // rho + v (DIM) + E
    g_halo.sendbuf_grad       = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap * grad_components);
    g_halo.recvbuf_grad       = (POINT_TYPE*)gpu_malloc(sizeof(POINT_TYPE) * total_cap * grad_components);

    g_halo.sendbuf_outer   = (unsigned char*)gpu_malloc(sizeof(unsigned char) * total_cap);
    g_halo.recvbuf_outer   = (unsigned char*)gpu_malloc(sizeof(unsigned char) * total_cap);
    g_halo.is_outer_layer  = (unsigned char*)gpu_malloc(sizeof(unsigned char) * total_cap);

    if (g_decomp.rank == 0) {
        printf("HALO: %d Cart neighbors, W_alloc=%d buckets, min_brick=%d, per_dir_capacity=%d, total_cap=%d\n",
               g_halo.n_neighbors, W_alloc, min_B, g_halo.per_dir_capacity, total_cap);
        fflush(stdout);
    }
#endif
}

void halo_free() {
#ifdef USE_MPI
    if (g_halo.export_indices) gpu_free(g_halo.export_indices);
    if (g_halo.sendbuf_seed)   gpu_free(g_halo.sendbuf_seed);
    if (g_halo.recvbuf_seed)   gpu_free(g_halo.recvbuf_seed);
    if (g_halo.sendbuf_rho)    gpu_free(g_halo.sendbuf_rho);
    if (g_halo.recvbuf_rho)    gpu_free(g_halo.recvbuf_rho);
    if (g_halo.sendbuf_v)      gpu_free(g_halo.sendbuf_v);
    if (g_halo.recvbuf_v)      gpu_free(g_halo.recvbuf_v);
    if (g_halo.sendbuf_E)      gpu_free(g_halo.sendbuf_E);
    if (g_halo.recvbuf_E)      gpu_free(g_halo.recvbuf_E);
    if (g_halo.sendbuf_vmesh)  gpu_free(g_halo.sendbuf_vmesh);
    if (g_halo.recvbuf_vmesh)  gpu_free(g_halo.recvbuf_vmesh);
    if (g_halo.sendbuf_grad)   gpu_free(g_halo.sendbuf_grad);
    if (g_halo.recvbuf_grad)   gpu_free(g_halo.recvbuf_grad);
    if (g_halo.sendbuf_outer)  gpu_free(g_halo.sendbuf_outer);
    if (g_halo.recvbuf_outer)  gpu_free(g_halo.recvbuf_outer);
    if (g_halo.is_outer_layer) gpu_free(g_halo.is_outer_layer);
    g_halo           = {};
    g_n_mpi_capacity = 0;
#endif
}

// ============================================================
// Export-list construction
// ============================================================

#ifdef USE_MPI
// encode a Cart direction (-1..+1)^3 into a unique tag in [1, 27]. Used to
// disambiguate messages from the same rank along different directions (e.g.
// np=2 1D periodic where +x and -x both point to the only other rank).
static inline int dir_tag(int dx, int dy, int dz) {
    return ((dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)) + 1;
}
#endif

// halo layer width in buckets; matches the periodic-ghost band thickness
static inline int halo_width_buckets(double buff, int N_grid) {
    double w = buff * (double)N_grid / (1.0 + 2.0 * buff);
    int    W = (int)std::ceil(w);
    return W < 1 ? 1 : W;
}

// true iff local cell at bucket (bx,by,bz) lies in the W-thick boundary layer
// facing direction (dx,dy,dz)
static inline bool cell_in_layer(int bx, int by, int bz, int dx, int dy, int dz,
                                 const int b0[3], const int b1[3], int W) {
    auto axis_match = [W](int b, int d, int lo, int hi) -> bool {
        if (d == 0)  return (b >= lo && b < hi);
        if (d == +1) return (b >= hi - W && b < hi);
        return (b >= lo && b < lo + W);  // d == -1
    };
    return axis_match(bx, dx, b0[0], b1[0])
        && axis_match(by, dy, b0[1], b1[1])
        && axis_match(bz, dz, b0[2], b1[2]);
}

// true iff the cell sits in the deepest (W-1)-th sub-layer along every non-zero axis of
// the direction. In sender coords this is the cell farthest from the boundary with the
// receiver; in receiver coords it lands at the outer edge of the imported halo band.
static inline bool cell_in_outermost_layer(int bx, int by, int bz, int dx, int dy, int dz,
                                           const int b0[3], const int b1[3], int W) {
    auto axis_match = [W](int b, int d, int lo, int hi) -> bool {
        if (d == 0)  return (b >= lo && b < hi);
        if (d == +1) return b == hi - W;
        return b == lo + W - 1;  // d == -1
    };
    return axis_match(bx, dx, b0[0], b1[0])
        && axis_match(by, dy, b0[1], b1[1])
        && axis_match(bz, dz, b0[2], b1[2]);
}

void halo_remap_export_indices(const unsigned int* inv_gather, int n_local) {
#ifndef USE_MPI
    (void)inv_gather; (void)n_local;
    return;
#else
    if (g_halo.n_neighbors == 0) return;
    const int pdc = g_halo.per_dir_capacity;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        for (int j = 0; j < g_halo.send_count[n]; j++) {
            const int slot  = n * pdc + j;
            const int old_k = g_halo.export_indices[slot];
            if (old_k < 0 || old_k >= n_local) continue;
            g_halo.export_indices[slot] = (int)inv_gather[old_k];
        }
    }
#endif
}

int halo_default_width(double buff) {
#ifdef USE_MPI
    return halo_width_buckets(buff, g_decomp.N_grid_global);
#else
    (void)buff;
    return 0;
#endif
}

void halo_build_exports(const POINT_TYPE* local_seeds, int n_local, double buff, int W_in) {
    g_halo.n_mpi_ghosts = 0;
    for (int n = 0; n < HALO_MAX_NEIGHBORS; n++) {
        g_halo.send_count[n]   = 0;
        g_halo.recv_count[n]   = 0;
        g_halo.ghost_offset[n] = 0;
    }
    g_halo.ghost_offset[HALO_MAX_NEIGHBORS] = 0;
    if (g_halo.n_neighbors == 0) return;

#ifdef USE_MPI
    const int N_grid = g_decomp.N_grid_global;
    const int b0[3]  = {g_decomp.b0[0], g_decomp.b0[1], g_decomp.b0[2]};
    const int b1[3]  = {g_decomp.b1[0], g_decomp.b1[1], g_decomp.b1[2]};
    const int W      = (W_in > 0) ? W_in : halo_width_buckets(buff, N_grid);

    // pass 1: count qualifying cells per direction
    for (int k = 0; k < n_local; k++) {
        double px = local_seeds[k].x;
        double py = local_seeds[k].y;
#ifdef dim_3D
        double pz = local_seeds[k].z;
#else
        double pz = 0.0;
#endif
        int bx, by, bz;
        decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);
        for (int n = 0; n < g_halo.n_neighbors; n++) {
            int dx = g_halo.neighbor_dirs[n][0];
            int dy = g_halo.neighbor_dirs[n][1];
            int dz = g_halo.neighbor_dirs[n][2];
            if (cell_in_layer(bx, by, bz, dx, dy, dz, b0, b1, W)) g_halo.send_count[n]++;
        }
    }

    for (int n = 0; n < g_halo.n_neighbors; n++) {
        if (g_halo.send_count[n] > g_halo.per_dir_capacity) {
            fprintf(stderr,
                    "[rank %d] HALO: send_count[%d]=%d > per_dir_capacity=%d (n_local=%d).\n",
                    g_decomp.rank, n, g_halo.send_count[n], g_halo.per_dir_capacity, n_local);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // pass 2: fill export_indices + sendbuf_outer in per-direction chunks of per_dir_capacity slots
    int idx[HALO_MAX_NEIGHBORS] = {0};
    for (int k = 0; k < n_local; k++) {
        double px = local_seeds[k].x;
        double py = local_seeds[k].y;
#ifdef dim_3D
        double pz = local_seeds[k].z;
#else
        double pz = 0.0;
#endif
        int bx, by, bz;
        decomp_bucket_of_point(px, py, pz, N_grid, buff, &bx, &by, &bz);
        for (int n = 0; n < g_halo.n_neighbors; n++) {
            int dx = g_halo.neighbor_dirs[n][0];
            int dy = g_halo.neighbor_dirs[n][1];
            int dz = g_halo.neighbor_dirs[n][2];
            if (cell_in_layer(bx, by, bz, dx, dy, dz, b0, b1, W)) {
                int slot                       = n * g_halo.per_dir_capacity + idx[n];
                g_halo.export_indices[slot]    = k;
                g_halo.sendbuf_outer[slot]     =
                    cell_in_outermost_layer(bx, by, bz, dx, dy, dz, b0, b1, W) ? 1 : 0;
                idx[n]++;
            }
        }
    }

    // exchange counts
    MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
    int         n_reqs = 0;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int dx = g_halo.neighbor_dirs[n][0];
        int dy = g_halo.neighbor_dirs[n][1];
        int dz = g_halo.neighbor_dirs[n][2];
        int peer = g_halo.neighbor_ranks[n];
        MPI_Isend(&g_halo.send_count[n], 1, MPI_INT, peer, dir_tag(dx, dy, dz),    g_halo.comm, &reqs[n_reqs++]);
        MPI_Irecv(&g_halo.recv_count[n], 1, MPI_INT, peer, dir_tag(-dx, -dy, -dz), g_halo.comm, &reqs[n_reqs++]);
    }
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    // cumulative ghost offsets
    int sum = 0;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        g_halo.ghost_offset[n] = sum;
        sum += g_halo.recv_count[n];
    }
    g_halo.ghost_offset[g_halo.n_neighbors] = sum;
    g_halo.n_mpi_ghosts                     = sum;

    if (sum > g_halo.n_mpi_capacity) {
        fprintf(stderr,
                "[rank %d] HALO: total recv (%d) > n_mpi_capacity (%d).\n",
                g_decomp.rank, sum, g_halo.n_mpi_capacity);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#else
    (void)local_seeds; (void)n_local; (void)buff; (void)W_in;
#endif
}

// ============================================================
// Exchange routines
// ============================================================

// per-quantity tag offset disambiguates the multiple Isend/Irecv pairs to the same
// peer along the same direction (e.g. np=2 1D periodic Cart)
enum HaloMsgKind { MSG_SEED = 0, MSG_RHO = 1, MSG_V = 2, MSG_E = 3, MSG_VMESH = 4, MSG_GRAD = 5,
                   MSG_OUTER = 6, MSG_KIND_STRIDE = 100 };

#ifdef USE_MPI
static inline int msg_tag(int dx, int dy, int dz, HaloMsgKind kind) {
    return dir_tag(dx, dy, dz) + (int)kind * MSG_KIND_STRIDE;
}
#endif

void halo_exchange_initial(VMesh* mesh, hydro::primvars* primvar, POINT_TYPE* pts, int pts_mpi_base) {
#ifndef USE_MPI
    (void)mesh; (void)primvar; (void)pts; (void)pts_mpi_base;
    return;
#else
    if (g_halo.n_neighbors == 0 || g_halo.n_mpi_ghosts == 0) return;

    const int n_hydro = (int)mesh->n_hydro;
    const int pdc     = g_halo.per_dir_capacity;

    // pack send buffers. seeds carry the periodic-wrap shift; primvars/v_mesh
    // are values and unaffected.
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        const double sx = g_halo.neighbor_shift[n][0];
        const double sy = g_halo.neighbor_shift[n][1];
#ifdef dim_3D
        const double sz = g_halo.neighbor_shift[n][2];
#endif
        for (int j = 0; j < g_halo.send_count[n]; j++) {
            int        k = g_halo.export_indices[n * pdc + j];
            int        s = n * pdc + j;
            POINT_TYPE p = pts[k];
            p.x += sx;
            p.y += sy;
#ifdef dim_3D
            p.z += sz;
#endif
            g_halo.sendbuf_seed[s] = p;
            g_halo.sendbuf_rho[s]  = primvar->rho[k];
            g_halo.sendbuf_v[s]    = primvar->v[k];
            g_halo.sendbuf_E[s]    = primvar->E[k];
#ifdef MOVING_MESH
            g_halo.sendbuf_vmesh[s] = mesh->v_mesh[k];
#else
            (void)mesh;
#endif
        }
    }

    // post Isend/Irecv per neighbor for each quantity
    int         n_reqs = 0;
    MPI_Request reqs[2 * 6 * HALO_MAX_NEIGHBORS];

    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int dx = g_halo.neighbor_dirs[n][0], dy = g_halo.neighbor_dirs[n][1], dz = g_halo.neighbor_dirs[n][2];
        int peer  = g_halo.neighbor_ranks[n];
        int sc    = g_halo.send_count[n], rc = g_halo.recv_count[n];
        int s_off = n * pdc;

        if (sc > 0) MPI_Isend(&g_halo.sendbuf_seed[s_off], sc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_SEED), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_seed[s_off], rc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_SEED), g_halo.comm, &reqs[n_reqs++]);
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_rho[s_off], sc, MPI_DOUBLE, peer,
                              msg_tag(dx, dy, dz, MSG_RHO), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_rho[s_off], rc, MPI_DOUBLE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_RHO), g_halo.comm, &reqs[n_reqs++]);
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_v[s_off], sc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_V), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_v[s_off], rc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_V), g_halo.comm, &reqs[n_reqs++]);
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_E[s_off], sc, MPI_DOUBLE, peer,
                              msg_tag(dx, dy, dz, MSG_E), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_E[s_off], rc, MPI_DOUBLE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_E), g_halo.comm, &reqs[n_reqs++]);
#ifdef MOVING_MESH
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_vmesh[s_off], sc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_VMESH), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_vmesh[s_off], rc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_VMESH), g_halo.comm, &reqs[n_reqs++]);
#endif
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_outer[s_off], sc, MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_OUTER), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_outer[s_off], rc, MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_OUTER), g_halo.comm, &reqs[n_reqs++]);
    }
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    // unpack into pts (KNN input), mesh->seeds, primvar, v_mesh, is_outer_layer.
    // ghost_ids is written by the caller after periodic ghosts (n_pgh unknown here).
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int s_off = n * pdc;
        int g_off = g_halo.ghost_offset[n];
        for (int j = 0; j < g_halo.recv_count[n]; j++) {
            int        slot  = g_off + j;
            int        ext_k = n_hydro + slot;
            int        pts_k = pts_mpi_base + slot;
            POINT_TYPE p     = g_halo.recvbuf_seed[s_off + j];

            pts[pts_k] = p;
#ifdef dim_3D
            mesh->seeds[ext_k] = double3{p.x, p.y, p.z};
#else
            mesh->seeds[ext_k] = double3{p.x, p.y, 0.0};
#endif
            primvar->rho[ext_k] = g_halo.recvbuf_rho[s_off + j];
            primvar->v[ext_k]   = g_halo.recvbuf_v[s_off + j];
            primvar->E[ext_k]   = g_halo.recvbuf_E[s_off + j];
#ifdef MOVING_MESH
            mesh->v_mesh[ext_k] = g_halo.recvbuf_vmesh[s_off + j];
#endif
            g_halo.is_outer_layer[slot] = g_halo.recvbuf_outer[s_off + j];
        }
    }
#endif
}

void halo_exchange_primvars(VMesh* mesh, hydro::primvars* primvar) {
#ifndef USE_MPI
    (void)mesh; (void)primvar;
    return;
#else
    if (g_halo.n_neighbors == 0 || g_halo.n_mpi_ghosts == 0) return;
    const int pdc = g_halo.per_dir_capacity;

    for (int n = 0; n < g_halo.n_neighbors; n++) {
        for (int j = 0; j < g_halo.send_count[n]; j++) {
            int k = g_halo.export_indices[n * pdc + j];
            int s = n * pdc + j;
            g_halo.sendbuf_rho[s] = primvar->rho[k];
            g_halo.sendbuf_v[s]   = primvar->v[k];
            g_halo.sendbuf_E[s]   = primvar->E[k];
        }
    }

    MPI_Request reqs[2 * 3 * HALO_MAX_NEIGHBORS];
    int         n_reqs = 0;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int dx = g_halo.neighbor_dirs[n][0], dy = g_halo.neighbor_dirs[n][1], dz = g_halo.neighbor_dirs[n][2];
        int peer = g_halo.neighbor_ranks[n];
        int sc = g_halo.send_count[n], rc = g_halo.recv_count[n];
        int s_off = n * pdc;
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_rho[s_off], sc, MPI_DOUBLE, peer,
                              msg_tag(dx, dy, dz, MSG_RHO), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_rho[s_off], rc, MPI_DOUBLE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_RHO), g_halo.comm, &reqs[n_reqs++]);
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_v[s_off], sc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_V), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_v[s_off], rc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_V), g_halo.comm, &reqs[n_reqs++]);
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_E[s_off], sc, MPI_DOUBLE, peer,
                              msg_tag(dx, dy, dz, MSG_E), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_E[s_off], rc, MPI_DOUBLE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_E), g_halo.comm, &reqs[n_reqs++]);
    }
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    const int n_hydro = (int)mesh->n_hydro;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int s_off = n * pdc;
        int g_off = g_halo.ghost_offset[n];
        for (int j = 0; j < g_halo.recv_count[n]; j++) {
            int ext_k = n_hydro + g_off + j;
            primvar->rho[ext_k] = g_halo.recvbuf_rho[s_off + j];
            primvar->v[ext_k]   = g_halo.recvbuf_v[s_off + j];
            primvar->E[ext_k]   = g_halo.recvbuf_E[s_off + j];
        }
    }
#endif
}

void halo_exchange_gradients(VMesh* mesh, gradients::PrimGradients* grads) {
#ifndef USE_MPI
    (void)mesh; (void)grads;
    return;
#else
    if (g_halo.n_neighbors == 0 || g_halo.n_mpi_ghosts == 0) return;
    const int pdc = g_halo.per_dir_capacity;

    // per-cell record: (3 + DIMENSION) POINT_TYPE-sized components — rho_grad, v_grad (DIM), E_grad
    const int N_COMP = 3 + DIMENSION;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        for (int j = 0; j < g_halo.send_count[n]; j++) {
            int k = g_halo.export_indices[n * pdc + j];
            int s = (n * pdc + j) * N_COMP;
            int c = 0;
            g_halo.sendbuf_grad[s + c++] = grads->rho[k];
            g_halo.sendbuf_grad[s + c++] = grads->vx[k];
            g_halo.sendbuf_grad[s + c++] = grads->vy[k];
#ifdef dim_3D
            g_halo.sendbuf_grad[s + c++] = grads->vz[k];
#endif
            g_halo.sendbuf_grad[s + c++] = grads->E[k];
        }
    }

    MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
    int         n_reqs = 0;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int dx = g_halo.neighbor_dirs[n][0], dy = g_halo.neighbor_dirs[n][1], dz = g_halo.neighbor_dirs[n][2];
        int peer = g_halo.neighbor_ranks[n];
        int sc = g_halo.send_count[n], rc = g_halo.recv_count[n];
        int s_off = n * pdc * N_COMP;
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_grad[s_off], sc * N_COMP * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_GRAD), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_grad[s_off], rc * N_COMP * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_GRAD), g_halo.comm, &reqs[n_reqs++]);
    }
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    const int n_hydro = (int)mesh->n_hydro;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        int g_off = g_halo.ghost_offset[n];
        for (int j = 0; j < g_halo.recv_count[n]; j++) {
            int ext_k = n_hydro + g_off + j;
            int s     = (n * pdc + j) * N_COMP;
            int c     = 0;
            grads->rho[ext_k] = g_halo.recvbuf_grad[s + c++];
            grads->vx[ext_k]  = g_halo.recvbuf_grad[s + c++];
            grads->vy[ext_k]  = g_halo.recvbuf_grad[s + c++];
#ifdef dim_3D
            grads->vz[ext_k] = g_halo.recvbuf_grad[s + c++];
#endif
            grads->E[ext_k] = g_halo.recvbuf_grad[s + c++];
        }
    }
#endif
}

void halo_exchange_vmesh(VMesh* mesh) {
#ifndef USE_MPI
    (void)mesh;
    return;
#else
#ifdef MOVING_MESH
    if (g_halo.n_neighbors == 0 || g_halo.n_mpi_ghosts == 0) return;
    const int pdc = g_halo.per_dir_capacity;

    for (int n = 0; n < g_halo.n_neighbors; n++) {
        for (int j = 0; j < g_halo.send_count[n]; j++) {
            const int k             = g_halo.export_indices[n * pdc + j];
            const int s             = n * pdc + j;
            g_halo.sendbuf_vmesh[s] = mesh->v_mesh[k];
        }
    }

    MPI_Request reqs[2 * HALO_MAX_NEIGHBORS];
    int         n_reqs = 0;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        const int dx = g_halo.neighbor_dirs[n][0], dy = g_halo.neighbor_dirs[n][1], dz = g_halo.neighbor_dirs[n][2];
        const int peer  = g_halo.neighbor_ranks[n];
        const int sc    = g_halo.send_count[n], rc = g_halo.recv_count[n];
        const int s_off = n * pdc;
        if (sc > 0) MPI_Isend(&g_halo.sendbuf_vmesh[s_off], sc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(dx, dy, dz, MSG_VMESH), g_halo.comm, &reqs[n_reqs++]);
        if (rc > 0) MPI_Irecv(&g_halo.recvbuf_vmesh[s_off], rc * (int)sizeof(POINT_TYPE), MPI_BYTE, peer,
                              msg_tag(-dx, -dy, -dz, MSG_VMESH), g_halo.comm, &reqs[n_reqs++]);
    }
    MPI_Waitall(n_reqs, reqs, MPI_STATUSES_IGNORE);

    const int n_hydro = (int)mesh->n_hydro;
    for (int n = 0; n < g_halo.n_neighbors; n++) {
        const int s_off = n * pdc;
        const int g_off = g_halo.ghost_offset[n];
        for (int j = 0; j < g_halo.recv_count[n]; j++) {
            const int ext_k     = n_hydro + g_off + j;
            mesh->v_mesh[ext_k] = g_halo.recvbuf_vmesh[s_off + j];
        }
    }
#else
    (void)mesh;
#endif
#endif
}

void halo_dt_allreduce(double* dt) {
#ifdef USE_MPI
    double local = *dt;
    MPI_Allreduce(&local, dt, 1, MPI_DOUBLE, MPI_MIN, g_halo.comm);
#else
    (void)dt;
#endif
}

}  // namespace proteus_mpi
