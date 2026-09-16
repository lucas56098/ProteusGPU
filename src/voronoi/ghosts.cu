
// periodic ghost seeds (internal.h)

namespace voronoi {

#ifdef CPU_DEBUG
    static uint64_t cpu_generate_periodic_ghosts(uint64_t          n_hydro,
                                                 const POINT_TYPE* pts_data,
                                                 POINT_TYPE*       pts,
                                                 uint64_t*         original_ids,
                                                 double            buff_val,
                                                 int               wx,
                                                 int               wy,
                                                 int               wz);
#endif
    HD static inline bool
    ghost_box_contains(POINT_TYPE pt, double xa, double xb, double ya, double yb, double za = 0.0, double zb = 1.0);
    static inline void append_ghost_copy(POINT_TYPE*     pts,
                                         uint64_t        index,
                                         uint64_t*       n_ghosts,
                                         const uint64_t* n_hydro,
                                         uint64_t*       original_ids,
                                         double          shift_x,
                                         double          shift_y,
                                         double          shift_z = 0.0);

#ifndef CPU_DEBUG
    static uint64_t launch_periodic_ghost_kernel(uint64_t          n_hydro,
                                                 const POINT_TYPE* pts_data,
                                                 POINT_TYPE*       pts,
                                                 uint64_t*         original_ids,
                                                 double            buff_val,
                                                 int               wx,
                                                 int               wy,
                                                 int               wz);
    GLOBAL void     kernel_generate_ghosts(uint64_t n_hydro,
                                           const POINT_TYPE* __restrict__ pts_data,
                                           POINT_TYPE* __restrict__ pts,
                                           uint64_t* __restrict__ original_ids,
                                           int* __restrict__ d_ghost_count,
                                           double buff_val,
                                           int    wx,
                                           int    wy,
                                           int    wz);
#endif

    // copies the seeds into pts and adds a shifted copy of the ones near a border
    uint64_t regenerate_periodic_ghosts(
        uint64_t n_hydro, const POINT_TYPE* pts_data, POINT_TYPE* pts, uint64_t* original_ids, double buff_val) {
        // only on axes this rank spans alone, else the copies come from the halo
        const int wx = (proteus_mpi::decomp.dims[0] == 1) ? 1 : 0;
        const int wy = (proteus_mpi::decomp.dims[1] == 1) ? 1 : 0;
        const int wz = (proteus_mpi::decomp.dims[2] == 1) ? 1 : 0;

#ifndef CPU_DEBUG
        return launch_periodic_ghost_kernel(n_hydro, pts_data, pts, original_ids, buff_val, wx, wy, wz);
#else
        return cpu_generate_periodic_ghosts(n_hydro, pts_data, pts, original_ids, buff_val, wx, wy, wz);
#endif
    }

#ifdef CPU_DEBUG
    // CPU version
    static uint64_t cpu_generate_periodic_ghosts(uint64_t          n_hydro,
                                                 const POINT_TYPE* pts_data,
                                                 POINT_TYPE*       pts,
                                                 uint64_t*         original_ids,
                                                 double            buff_val,
                                                 int               wx,
                                                 int               wy,
                                                 int               wz) {
        uint64_t n_ghosts = 0;
        for (uint64_t i = 0; i < n_hydro; i++) {
            pts[i] = pts_data[i];

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

                        // band this shift copies from
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
#endif

#ifndef CPU_DEBUG
    static uint64_t launch_periodic_ghost_kernel(uint64_t          n_hydro,
                                                 const POINT_TYPE* pts_data,
                                                 POINT_TYPE*       pts,
                                                 uint64_t*         original_ids,
                                                 double            buff_val,
                                                 int               wx,
                                                 int               wy,
                                                 int               wz) {
        int* d_ghost_count = (int*)gpu_malloc(sizeof(int));
        gpu_memset(d_ghost_count, 0, sizeof(int));

        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = ((int)n_hydro + tpb - 1) / tpb;
        kernel_generate_ghosts<<<blocks, tpb>>>(
            n_hydro, pts_data, pts, original_ids, d_ghost_count, buff_val, wx, wy, wz);
        GPU_SYNC();

        const uint64_t n_ghosts = (uint64_t)(*d_ghost_count);
        gpu_free(d_ghost_count);
        return n_ghosts;
    }
#endif

    // is the seed in the band
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

    // one shifted copy behind the cells
    static inline void append_ghost_copy(POINT_TYPE*     pts,
                                         uint64_t        index,
                                         uint64_t*       n_ghosts,
                                         const uint64_t* n_hydro,
                                         uint64_t*       original_ids,
                                         double          shift_x,
                                         double          shift_y,
                                         double          shift_z) {
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

#ifndef CPU_DEBUG

    // GPU version: count the copies of a seed, take the slots, write them
    GLOBAL void kernel_generate_ghosts(uint64_t n_hydro,
                                       const POINT_TYPE* __restrict__ pts_data,
                                       POINT_TYPE* __restrict__ pts,
                                       uint64_t* __restrict__ original_ids,
                                       int* __restrict__ d_ghost_count,
                                       double buff_val,
                                       int    wx,
                                       int    wy,
                                       int    wz) {
        const uint64_t i      = blockIdx.x * blockDim.x + threadIdx.x;
        const bool     active = (i < n_hydro);

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

        // count first
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

        // one atomic per warp, each thread takes its slots inside the warp block
        const unsigned full_mask = 0xffffffffu;
        int            s         = my_count;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            int t = __shfl_up_sync(full_mask, s, d);
            if ((int)(threadIdx.x & 31) >= d) s += t;
        }
        const int warp_total = __shfl_sync(full_mask, s, 31);
        const int my_excl    = s - my_count;

        int warp_base = 0;
        if ((threadIdx.x & 31) == 0 && warp_total > 0) { warp_base = portable_atomicAdd(d_ghost_count, warp_total); }
        warp_base = __shfl_sync(full_mask, warp_base, 0);

        if (!active || my_count == 0) return;

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

#endif

} // namespace voronoi
