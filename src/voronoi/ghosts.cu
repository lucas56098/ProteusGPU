namespace voronoi {

    // ---- forward declarations ----
    static hsize_t cpu_generate_periodic_ghosts(hsize_t           n_hydro,
                                                const POINT_TYPE* pts_data,
                                                POINT_TYPE*       pts,
                                                hsize_t*          original_ids,
                                                double            buff_val,
                                                int               wx,
                                                int               wy,
                                                int               wz);
    HD static inline bool
    ghost_box_contains(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za = 0.0, double zb = 1.0);
    static inline void append_ghost_copy(POINT_TYPE*    pts,
                                         hsize_t        index,
                                         hsize_t*       n_ghosts,
                                         const hsize_t* n_hydro,
                                         hsize_t*       original_ids,
                                         double         shift_x,
                                         double         shift_y,
                                         double         shift_z = 0.0);

#ifndef CPU_DEBUG
    static hsize_t launch_periodic_ghost_kernel(hsize_t           n_hydro,
                                                const POINT_TYPE* pts_data,
                                                POINT_TYPE*       pts,
                                                hsize_t*          original_ids,
                                                double            buff_val,
                                                int               wx,
                                                int               wy,
                                                int               wz);
    GLOBAL void    kernel_generate_ghosts(hsize_t n_hydro,
                                          const POINT_TYPE* __restrict__ pts_data,
                                          POINT_TYPE* __restrict__ pts,
                                          hsize_t* __restrict__ original_ids,
                                          int* __restrict__ d_ghost_count,
                                          double buff_val,
                                          int    wx,
                                          int    wy,
                                          int    wz);
#endif

    // ============================================================
    // Main routines
    // ============================================================

    // emit periodic ghost copies of boundary cells.
    //
    // Periodic-image generation only fires along an axis where the MPI decomposition has
    // exactly 1 rank in that direction. For decomposed axes the halo exchange already
    // carries the periodic wrap (see halo_internal.cu neighbor_shift), so a local periodic
    // ghost would land outside this rank's brick and never be used.
    hsize_t regenerate_periodic_ghosts(
        hsize_t n_hydro, const POINT_TYPE* pts_data, POINT_TYPE* pts, hsize_t* original_ids, double buff_val) {
        // wrap flags per axis: 1 if undecomposed (single rank), else 0
        const int wx = (proteus_mpi::decomp.dims[0] == 1) ? 1 : 0;
        const int wy = (proteus_mpi::decomp.dims[1] == 1) ? 1 : 0;
        const int wz = (proteus_mpi::decomp.dims[2] == 1) ? 1 : 0;

#ifndef CPU_DEBUG
        return launch_periodic_ghost_kernel(n_hydro, pts_data, pts, original_ids, buff_val, wx, wy, wz);
#else
        return cpu_generate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff_val, wx, wy, wz);
#endif
    }

    // ============================================================
    // Helpers
    // ============================================================

    // CPU path: per-real-cell loop over the 3^d - 1 periodic offsets
    static hsize_t cpu_generate_periodic_ghosts(hsize_t           n_hydro,
                                                const POINT_TYPE* pts_data,
                                                POINT_TYPE*       pts,
                                                hsize_t*          original_ids,
                                                double            buff_val,
                                                int               wx,
                                                int               wy,
                                                int               wz) {
        hsize_t n_ghosts = 0;
        for (hsize_t i = 0; i < n_hydro; i++) {
            // copy the real cell into pts[]
            pts[i] = pts_data[i];

            // for each periodic offset (sx, sy, sz), emit a ghost if the cell sits in the strip
            for (int sx = -wx; sx <= wx; sx++) {
                for (int sy = -wy; sy <= wy; sy++) {
#ifdef dim_3D
                    for (int sz = -wz; sz <= wz; sz++) {
#else
                    {
                        int sz = 0;
                        (void)wz;
#endif
                        if (sx == 0 && sy == 0 && sz == 0) continue;

                        // strip extents for this offset (one buff-thick band per +/- axis)
                        double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                        double xb = (sx == 1) ? buff_val : 1.0;
                        double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                        double yb = (sy == 1) ? buff_val : 1.0;
                        double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                        double zb = (sz == 1) ? buff_val : 1.0;
                        if (ghost_box_contains(pts[i], xa, xb, ya, yb, za, zb)) {
                            append_ghost_copy(
                                pts, i, &n_ghosts, &n_hydro, original_ids, (double)sx, (double)sy, (double)sz);
                        }
                    }
                }
            }
        }
        return n_ghosts;
    }

#ifndef CPU_DEBUG
    // GPU path: launch the warp-aggregated ghost kernel and read back the produced count
    static hsize_t launch_periodic_ghost_kernel(hsize_t           n_hydro,
                                                const POINT_TYPE* pts_data,
                                                POINT_TYPE*       pts,
                                                hsize_t*          original_ids,
                                                double            buff_val,
                                                int               wx,
                                                int               wy,
                                                int               wz) {
        // device-side counter for ghost slots claimed by warps
        int* d_ghost_count = (int*)gpu_malloc(sizeof(int));
        gpu_memset(d_ghost_count, 0, sizeof(int));

        // launch one thread per real cell
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_generate_ghosts<<<blocks, tpb>>>(
            n_hydro, pts_data, pts, original_ids, d_ghost_count, buff_val, wx, wy, wz);
        GPU_SYNC();

        // read the total ghost count back and free the counter
        const hsize_t n_ghosts = (hsize_t)(*d_ghost_count);
        gpu_free(d_ghost_count);
        return n_ghosts;
    }
#endif

    // is pt inside the half-open box (xa, xb) x (ya, yb) (x (za, zb))?
    HD static inline bool
    ghost_box_contains(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za, double zb) {
#ifdef dim_2D
        (void)za;
        (void)zb;
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb);
#else
        return (pt.x > xa && pt.x < xb) && (pt.y > ya && pt.y < yb) && (pt.z > za && pt.z < zb);
#endif
    }

    // write one shifted copy of cell index into pts[n_hydro + n_ghosts]; bump n_ghosts
    static inline void append_ghost_copy(POINT_TYPE*    pts,
                                         hsize_t        index,
                                         hsize_t*       n_ghosts,
                                         const hsize_t* n_hydro,
                                         hsize_t*       original_ids,
                                         double         shift_x,
                                         double         shift_y,
                                         double         shift_z) {
        POINT_TYPE pt;
        pt.x = pts[index].x + shift_x;
        pt.y = pts[index].y + shift_y;
#ifdef dim_3D
        pt.z = pts[index].z + shift_z;
#else
        (void)shift_z;
#endif

        pts[(*n_hydro) + (*n_ghosts)] = pt;
        original_ids[*n_ghosts]       = index;
        (*n_ghosts)++;
    }

    // ============================================================
    // CUDA kernels
    // ============================================================
#ifndef CPU_DEBUG

    // warp-aggregated ghost generation: each warp computes its members' ghost counts,
    // prefix-sums them across lanes, then a single atomicAdd by lane 0 claims a contiguous
    // slot range. Each lane writes its ghosts at known offsets within that range. Replaces
    // up to 7 atomics per thread (3D corner cell) with 1 atomic per warp — critical when
    // post-spatial-sort threads in a warp are spatially clustered.
    GLOBAL void kernel_generate_ghosts(hsize_t n_hydro,
                                       const POINT_TYPE* __restrict__ pts_data,
                                       POINT_TYPE* __restrict__ pts,
                                       hsize_t* __restrict__ original_ids,
                                       int* __restrict__ d_ghost_count,
                                       double buff_val,
                                       int    wx,
                                       int    wy,
                                       int    wz) {
        const hsize_t i      = blockIdx.x * blockDim.x + threadIdx.x;
        const bool    active = (i < n_hydro);

        // copy real cell into scratch_pts (every thread does this; out-of-range threads
        // zero pi so the geometry test stays well-defined for warp prefix-sum)
        POINT_TYPE pi;
        if (active) {
            pts[i] = pts_data[i];
            pi     = pts[i];
        } else {
            pi.x = 0.0;
            pi.y = 0.0;
#ifdef dim_3D
            pi.z = 0.0;
#endif
        }

        // pass 1: count ghosts this thread will produce (0..7 in 3D, 0..3 in 2D)
        int my_count = 0;
        if (active) {
            for (int sx = -wx; sx <= wx; sx++) {
                for (int sy = -wy; sy <= wy; sy++) {
#ifdef dim_3D
                    for (int sz = -wz; sz <= wz; sz++) {
#else
                    {
                        int sz = 0;
                        (void)wz;
#endif
                        if (sx == 0 && sy == 0 && sz == 0) continue;

                        double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                        double xb = (sx == 1) ? buff_val : 1.0;
                        double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                        double yb = (sy == 1) ? buff_val : 1.0;
                        double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                        double zb = (sz == 1) ? buff_val : 1.0;

                        if (ghost_box_contains(pi, xa, xb, ya, yb, za, zb)) my_count++;
                    }
                }
            }
        }

        // warp inclusive prefix sum over my_count (Kogge-Stone via __shfl_up_sync).
        // All 32 lanes must participate; inactive threads contribute my_count == 0.
        const unsigned full_mask = 0xffffffffu;
        int            s         = my_count;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            int t = __shfl_up_sync(full_mask, s, d);
            if ((int)(threadIdx.x & 31) >= d) s += t;
        }
        const int warp_total = __shfl_sync(full_mask, s, 31);
        const int my_excl    = s - my_count;

        // one atomicAdd per warp to claim a contiguous slot range
        int warp_base = 0;
        if ((threadIdx.x & 31) == 0 && warp_total > 0) { warp_base = portable_atomicAdd(d_ghost_count, warp_total); }
        warp_base = __shfl_sync(full_mask, warp_base, 0);

        if (!active || my_count == 0) return;

        // pass 2: re-run the geometry tests and write ghosts at known slots
        // (recomputing is cheaper than spilling 7 directions into local memory)
        const int my_base   = warp_base + my_excl;
        int       n_written = 0;
        for (int sx = -wx; sx <= wx; sx++) {
            for (int sy = -wy; sy <= wy; sy++) {
#ifdef dim_3D
                for (int sz = -wz; sz <= wz; sz++) {
#else
                {
                    int sz = 0;
#endif
                    if (sx == 0 && sy == 0 && sz == 0) continue;

                    double xa = (sx == 1) ? 0.0 : (sx == -1) ? 1.0 - buff_val : 0.0;
                    double xb = (sx == 1) ? buff_val : 1.0;
                    double ya = (sy == 1) ? 0.0 : (sy == -1) ? 1.0 - buff_val : 0.0;
                    double yb = (sy == 1) ? buff_val : 1.0;
                    double za = (sz == 1) ? 0.0 : (sz == -1) ? 1.0 - buff_val : 0.0;
                    double zb = (sz == 1) ? buff_val : 1.0;

                    if (ghost_box_contains(pi, xa, xb, ya, yb, za, zb)) {
                        const int  slot = my_base + n_written;
                        POINT_TYPE gpt;
                        gpt.x = pi.x + (double)sx;
                        gpt.y = pi.y + (double)sy;
#ifdef dim_3D
                        gpt.z = pi.z + (double)sz;
#endif
                        pts[n_hydro + slot] = gpt;
                        original_ids[slot]  = i;
                        n_written++;
                    }
                }
            }
        }
    }

#endif // !CPU_DEBUG

} // namespace voronoi
