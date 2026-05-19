#include "../global/allvars.h"
#include "../global/structs.h"
#include "../knn/knn.h"
#include "../mpi/halo.h"
#include "../mpi/mpi_compat.h"
#include "../profiler/profiler.h"
#include "cell.h"
#include "voronoi.h"

#include "cell.cu"
#include "geometry.cu"

#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>

namespace voronoi {

    // forward declarations
    template <int K, int MAX_P, int MAX_T>
    HD void compute_single_voronoi_cell(int                 k,
                                        int                 seed_id,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag);

#ifndef CPU_DEBUG
    GLOBAL void kernel_init_cell_status(int n, Status* stat);
    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count);
    GLOBAL void kernel_collect_failed_cells(int n, const Status* stat, int* failed_indices, int* failed_count);
    GLOBAL void kernel_compute_voronoi_cells_fast(int                n_hydro,
                                                  double*            d_stored_points,
                                                  const knn_problem* knn,
                                                  Status*            stat,
                                                  VMesh*             mesh,
                                                  hsize_t*           face_offset,
                                                  int*               overflow_flag);
    GLOBAL void kernel_compute_voronoi_cells_slow(int                n_failed,
                                                  const int*         failed_ks,
                                                  double*            d_stored_points,
                                                  const knn_problem* knn,
                                                  Status*            stat,
                                                  VMesh*             mesh,
                                                  hsize_t*           face_offset,
                                                  int*               overflow_flag);
#endif

    // ============================================================
    // Allocation
    // ============================================================

    VMesh* allocate_mesh(hsize_t n_hydro) {
        // n_grow = max post-migration n_local. Periodic and MPI ghosts coexist in pts[] and ghost_ids,
        // and capacities scale with n_grow so all mesh-build buffers survive migration imbalance.
        // ext sizes the per-cell SoA arrays for n_grow + MPI ghost slots.
        const double  ghost_frac     = pow(1.0 + 2.0 * buff, (double)DIMENSION) - 1.0;
        const hsize_t n_grow         = (hsize_t)proteus_mpi::max_n_local((int)n_hydro);
        const hsize_t max_pgh        = (hsize_t)(2.0 * ghost_frac * n_grow) + 1;
        const hsize_t max_mpi_ghosts = (hsize_t)proteus_mpi::g_n_mpi_capacity;
        const hsize_t max_ghosts     = max_pgh + max_mpi_ghosts;
        const hsize_t total          = n_grow + max_ghosts;
        const hsize_t max_faces      = n_grow * _FACE_CAPACITY_MULT_;
        const hsize_t ext            = (hsize_t)proteus_mpi::alloc_per_cell_size((int)n_hydro);

        VMesh* mesh          = gpu_alloc<VMesh>(1);
        mesh->n_seeds        = 0;
        mesh->n_hydro        = n_hydro;
        mesh->num_faces      = 0;
        mesh->face_capacity  = max_faces;
        mesh->ghost_capacity = max_ghosts;
        mesh->total_capacity = total;
        mesh->buff           = buff;

        // per-cell — every n_hydro-indexed array is sized ext so it survives migration
        // growth and the MPI ghost band [n_hydro, n_hydro + n_mpi_ghosts)
        mesh->seeds       = gpu_calloc<double3>(ext);
        mesh->com         = gpu_calloc<double3>(ext);
        mesh->volumes     = gpu_calloc<double>(ext);
        mesh->face_counts = gpu_calloc<hsize_t>(ext);
        mesh->face_ptr    = gpu_calloc<hsize_t>(ext);
        mesh->cell_status = gpu_alloc<Status>(ext);
#ifdef MOVING_MESH
        mesh->v_mesh      = gpu_calloc<POINT_TYPE>(ext);
        mesh->old_volumes = gpu_calloc<double>(ext);
#endif

        // per-face
        mesh->neighbor_cell = gpu_alloc<int>(max_faces);
        mesh->face_area     = gpu_alloc<double>(max_faces);
#ifdef MOVING_MESH
        mesh->f_mid_local = gpu_alloc<double>(max_faces * (DIMENSION - 1));
#endif

        // ghost mapping
        mesh->ghost_ids = gpu_alloc<hsize_t>(max_ghosts);

        // index maps; ext-sized so permute_inplace's live↔scratch swap stays uniform-size
        mesh->real_sorted_ids  = gpu_alloc<unsigned int>(ext);
        mesh->sid_to_neighbor  = gpu_alloc<unsigned int>(total);
        mesh->cell_to_original = gpu_alloc<unsigned int>(ext);
        mesh->gather_perm      = gpu_alloc<unsigned int>(ext);
        mesh->orig_to_k_save   = gpu_alloc<unsigned int>(ext);
        for (hsize_t i = 0; i < n_hydro; i++)
            mesh->cell_to_original[i] = (unsigned int)i;

        // typed scratch pools — ext-sized (see above)
        mesh->scratch_uint   = gpu_alloc<unsigned int>(ext);
        mesh->scratch_double = gpu_alloc<double>(ext);
        mesh->scratch_point  = gpu_alloc<POINT_TYPE>(ext);

        // mesh-build scratch
        mesh->scratch_pts  = gpu_alloc<POINT_TYPE>(total);
        mesh->scratch_move = gpu_alloc<POINT_TYPE>(ext);

        // device counters
        mesh->d_real_counter = gpu_calloc<int>(1);

        // KNN cache
        mesh->knn = knn::init_once((int)n_hydro);

        // hint GPU-preferred placement for hot arrays
        gpu_advise_gpu_preferred(mesh->seeds, ext * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->com, n_hydro * sizeof(double3));
        gpu_advise_gpu_preferred(mesh->volumes, n_hydro * sizeof(double));
        gpu_advise_gpu_preferred(mesh->face_counts, n_hydro * sizeof(hsize_t));
        gpu_advise_gpu_preferred(mesh->face_ptr, n_hydro * sizeof(hsize_t));
        gpu_advise_gpu_preferred(mesh->cell_status, n_hydro * sizeof(Status));
        gpu_advise_gpu_preferred(mesh->neighbor_cell, max_faces * sizeof(int));
        gpu_advise_gpu_preferred(mesh->face_area, max_faces * sizeof(double));
        gpu_advise_gpu_preferred(mesh->real_sorted_ids, n_hydro * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->sid_to_neighbor, total * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->cell_to_original, n_hydro * sizeof(unsigned int));
        gpu_advise_gpu_preferred(mesh->gather_perm, n_hydro * sizeof(unsigned int));

        return mesh;
    }

    void free_mesh(VMesh* mesh) {
        if (!mesh) return;
        gpu_free(mesh->seeds);
        gpu_free(mesh->com);
        gpu_free(mesh->volumes);
        gpu_free(mesh->face_counts);
        gpu_free(mesh->face_ptr);
        gpu_free(mesh->cell_status);
#ifdef MOVING_MESH
        gpu_free(mesh->v_mesh);
        gpu_free(mesh->old_volumes);
        gpu_free(mesh->f_mid_local);
#endif
        gpu_free(mesh->neighbor_cell);
        gpu_free(mesh->face_area);
        gpu_free(mesh->ghost_ids);
        gpu_free(mesh->real_sorted_ids);
        gpu_free(mesh->sid_to_neighbor);
        gpu_free(mesh->cell_to_original);
        gpu_free(mesh->gather_perm);
        gpu_free(mesh->orig_to_k_save);
        gpu_free(mesh->scratch_uint);
        gpu_free(mesh->scratch_double);
        gpu_free(mesh->scratch_point);
        gpu_free(mesh->scratch_pts);
        gpu_free(mesh->scratch_move);
        gpu_free(mesh->d_real_counter);
        if (mesh->knn) { knn::knn_free(&mesh->knn); }
        gpu_free(mesh);
    }

    // ============================================================
    // Permutation pipeline
    // ============================================================

#ifndef CPU_DEBUG
    template <typename T> GLOBAL void kernel_gather(hsize_t n, const T* in, const unsigned int* perm, T* out) {
        hsize_t k = (hsize_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (k < n) out[k] = in[perm[k]];
    }

    GLOBAL void kernel_build_index_pass1_compact_reals(int                 n_total,
                                                       hsize_t             n_hydro,
                                                       const unsigned int* d_permutation,
                                                       unsigned int*       real_sorted_ids,
                                                       unsigned int*       sid_to_neighbor,
                                                       unsigned int*       orig_to_k,
                                                       int*                counter) {
        const int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= n_total) return;
        const unsigned int orig = d_permutation[sid];
        if ((hsize_t)orig < n_hydro) {
            const int k          = atomicAdd(counter, 1);
            real_sorted_ids[k]   = (unsigned int)sid;
            sid_to_neighbor[sid] = (unsigned int)k;
            orig_to_k[orig]      = (unsigned int)k;
        }
    }

    // stable-k variant of pass1: reads the saved orig_to_k mapping from iter 0 so that
    // the iter > 0 cell-to-k assignment matches iter 0 (primvar was already permuted then)
    GLOBAL void kernel_build_index_pass1_lookup(int                 n_total,
                                                hsize_t             n_hydro,
                                                const unsigned int* d_permutation,
                                                const unsigned int* orig_to_k_save,
                                                unsigned int*       real_sorted_ids,
                                                unsigned int*       sid_to_neighbor,
                                                unsigned int*       orig_to_k) {
        const int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= n_total) return;
        const unsigned int orig = d_permutation[sid];
        if ((hsize_t)orig < n_hydro) {
            const unsigned int k = orig_to_k_save[orig];
            real_sorted_ids[k]   = (unsigned int)sid;
            sid_to_neighbor[sid] = k;
            orig_to_k[orig]      = k; // populate scratch_uint for pass2 (periodic-ghost branch)
        }
    }

    GLOBAL void kernel_build_index_pass2_remap_ghosts(int                 n_total,
                                                      hsize_t             n_hydro,
                                                      const unsigned int* d_permutation,
                                                      const hsize_t*      ghost_ids,
                                                      const unsigned int* orig_to_k,
                                                      unsigned int*       sid_to_neighbor) {
        const int sid = blockIdx.x * blockDim.x + threadIdx.x;
        if (sid >= n_total) return;
        const unsigned int orig = d_permutation[sid];
        if ((hsize_t)orig >= n_hydro) {
            const hsize_t      g = (hsize_t)orig - n_hydro;
            const unsigned int v = (unsigned int)ghost_ids[g];
            // MPI ghost: ghost_ids holds the extended array index (>= n_hydro).
            // Periodic ghost: ghost_ids holds the source orig, mapped to its k via orig_to_k.
            sid_to_neighbor[sid] = (v >= (unsigned int)n_hydro) ? v : orig_to_k[v];
        }
    }
#endif

    // Build real_sorted_ids[k] -> sid and sid_to_neighbor[sid] -> k.
    // The pass-1 orig→k map is stashed in scratch_uint; pass 2 reads it.
    // iter == 0: atomic-counter pass1, saves the mapping in orig_to_k_save.
    // iter > 0: lookup pass1 from orig_to_k_save so primvar (permuted in iter 0) stays aligned.
    static void build_index_maps(VMesh* mesh, int iter) {
        const int     n_total = (int)mesh->n_seeds;
        const hsize_t n_hydro = mesh->n_hydro;

#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (n_total + tpb - 1) / tpb;

        if (iter == 0) {
            gpu_memset(mesh->d_real_counter, 0, sizeof(int));
            kernel_build_index_pass1_compact_reals<<<blocks, tpb>>>(n_total,
                                                                    n_hydro,
                                                                    mesh->knn->d_permutation,
                                                                    mesh->real_sorted_ids,
                                                                    mesh->sid_to_neighbor,
                                                                    mesh->scratch_uint,
                                                                    mesh->d_real_counter);
            GPU_LAUNCH_CHECK();
        } else {
            kernel_build_index_pass1_lookup<<<blocks, tpb>>>(n_total,
                                                             n_hydro,
                                                             mesh->knn->d_permutation,
                                                             mesh->orig_to_k_save,
                                                             mesh->real_sorted_ids,
                                                             mesh->sid_to_neighbor,
                                                             mesh->scratch_uint);
            GPU_LAUNCH_CHECK();
        }

        kernel_build_index_pass2_remap_ghosts<<<blocks, tpb>>>(
            n_total, n_hydro, mesh->knn->d_permutation, mesh->ghost_ids, mesh->scratch_uint, mesh->sid_to_neighbor);
        GPU_LAUNCH_CHECK();
        GPU_SYNC();

        if (iter == 0 && (hsize_t)*mesh->d_real_counter != n_hydro) {
            std::cerr << "VORONOI: build_index_maps: counted " << *mesh->d_real_counter
                      << " reals but n_hydro = " << n_hydro << ". Aborting." << std::endl;
            exit(EXIT_FAILURE);
        }
#else
        const unsigned int* dperm = mesh->knn->d_permutation;
        if (iter == 0) {
            unsigned int k = 0;
            for (int sid = 0; sid < n_total; sid++) {
                const unsigned int orig = dperm[sid];
                if ((hsize_t)orig < n_hydro) {
                    mesh->real_sorted_ids[k]   = (unsigned int)sid;
                    mesh->sid_to_neighbor[sid] = k;
                    mesh->scratch_uint[orig]   = k;
                    k++;
                }
            }
            if ((hsize_t)k != n_hydro) {
                std::cerr << "VORONOI: build_index_maps: counted " << k << " reals but n_hydro = " << n_hydro
                          << ". Aborting." << std::endl;
                exit(EXIT_FAILURE);
            }
        } else {
            for (int sid = 0; sid < n_total; sid++) {
                const unsigned int orig = dperm[sid];
                if ((hsize_t)orig < n_hydro) {
                    const unsigned int k       = mesh->orig_to_k_save[orig];
                    mesh->real_sorted_ids[k]   = (unsigned int)sid;
                    mesh->sid_to_neighbor[sid] = k;
                    mesh->scratch_uint[orig]   = k;
                }
            }
        }
        for (int sid = 0; sid < n_total; sid++) {
            const unsigned int orig = dperm[sid];
            if ((hsize_t)orig >= n_hydro) {
                const hsize_t      g = (hsize_t)orig - n_hydro;
                const unsigned int v = (unsigned int)mesh->ghost_ids[g];
                // MPI ghost: v is the extended array index. Periodic ghost: v is the source orig.
                mesh->sid_to_neighbor[sid] = (v >= (unsigned int)n_hydro) ? v : mesh->scratch_uint[v];
            }
        }
#endif
    }

    // gather_perm[new_k] = d_permutation[real_sorted_ids[new_k]]
    //                    = orig of new_k = old_k (because step N's k IS step N+1's input orig).
    static void compute_gather_perm(VMesh* mesh) {
        const hsize_t n = mesh->n_hydro;
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (int)((n + tpb - 1) / tpb);
        kernel_gather<unsigned int>
            <<<blocks, tpb>>>(n, mesh->knn->d_permutation, mesh->real_sorted_ids, mesh->gather_perm);
        GPU_LAUNCH_CHECK();
#else
        for (hsize_t k = 0; k < n; k++) {
            mesh->gather_perm[k] = mesh->knn->d_permutation[mesh->real_sorted_ids[k]];
        }
#endif
    }

    // Out-of-place gather then pointer swap.
    // Default-stream serialization handles ordering between back-to-back calls.
    // The permutation only touches [0, n); the MPI-ghost-slot region [n, ext) is copied
    // verbatim from live to scratch so it survives the pointer swap.
    template <typename T> static void permute_inplace(T*& live, T*& scratch, hsize_t n, const unsigned int* perm) {
        const hsize_t ext = (hsize_t)proteus_mpi::extended_size((int)n);
#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (int)((n + tpb - 1) / tpb);
        kernel_gather<T><<<blocks, tpb>>>(n, live, perm, scratch);
        GPU_LAUNCH_CHECK();
        if (ext > n) { gpu_memcpy(scratch + n, live + n, (ext - n) * sizeof(T)); }
#else
        for (hsize_t k = 0; k < n; k++)
            scratch[k] = live[perm[k]];
        for (hsize_t k = n; k < ext; k++)
            scratch[k] = live[k];
#endif
        std::swap(live, scratch);
    }

    static void permute_persistent_state(VMesh* mesh, hydro::primvars* primvar, hydro::primvars* primvar_aux) {
        const hsize_t       n    = mesh->n_hydro;
        const unsigned int* perm = mesh->gather_perm;

        permute_inplace(mesh->cell_to_original, mesh->scratch_uint, n, perm);

        if (primvar) {
            permute_inplace(primvar->rho, mesh->scratch_double, n, perm);
            permute_inplace(primvar->v, mesh->scratch_point, n, perm);
            permute_inplace(primvar->E, mesh->scratch_double, n, perm);
        }
        if (primvar_aux) {
            permute_inplace(primvar_aux->rho, mesh->scratch_double, n, perm);
            permute_inplace(primvar_aux->v, mesh->scratch_point, n, perm);
            permute_inplace(primvar_aux->E, mesh->scratch_double, n, perm);
        }
#ifdef MOVING_MESH
        permute_inplace(mesh->v_mesh, mesh->scratch_point, n, perm);
        permute_inplace(mesh->old_volumes, mesh->scratch_double, n, perm);
#endif

#ifndef CPU_DEBUG
        GPU_SYNC();
#endif
    }

    // ============================================================
    // Top-level mesh build
    // ============================================================

    void compute_mesh(VMesh*           mesh,
                      POINT_TYPE*      pts_data,
                      int              n_total,
                      hydro::primvars* primvar,
                      hydro::primvars* primvar_aux,
                      int              iter) {

        Profiler::StartTimer("KNN (par)");
        knn::prepare(mesh->knn, (const POINT_TYPE*)pts_data, n_total);
        Profiler::EndTimer("KNN (par)");

        if ((hsize_t)n_total > mesh->total_capacity) {
            std::cerr << "VORONOI: Error! point count " << n_total << " exceeds pre-allocated capacity "
                      << mesh->total_capacity << ". Increase ghost headroom." << std::endl;
            exit(EXIT_FAILURE);
        }

        mesh->n_seeds   = (hsize_t)n_total;
        mesh->num_faces = 0;

        Profiler::StartTimer("PERMUTE (par)");
        build_index_maps(mesh, iter);
        if (iter == 0) {
            // snapshot pass1's orig_to_k so iter > 0 can reproduce the same k assignment
            const hsize_t n_hydro_local = mesh->n_hydro;
            gpu_memcpy(mesh->orig_to_k_save, mesh->scratch_uint, n_hydro_local * sizeof(unsigned int));

            compute_gather_perm(mesh);
            permute_persistent_state(mesh, primvar, primvar_aux);
        }
        // iter > 0: primvar already in iter-0-k order; build_index_maps used the saved
        // mapping, so sid_to_neighbor is consistent without a second permute
        Profiler::EndTimer("PERMUTE (par)");

        Profiler::StartTimer("VORONOI (par)");

        const hsize_t n_hydro = mesh->n_hydro;
        gpu_memset(mesh->face_counts, 0, n_hydro * sizeof(hsize_t));
        gpu_memset(mesh->face_ptr, 0, n_hydro * sizeof(hsize_t));

#ifndef CPU_DEBUG
        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (int)((n_hydro + tpb - 1) / tpb);
        kernel_init_cell_status<<<blocks, tpb>>>((int)n_hydro, mesh->cell_status);
        GPU_LAUNCH_CHECK();
#else
        for (hsize_t i = 0; i < n_hydro; i++)
            mesh->cell_status[i] = security_radius_not_reached;
#endif

        compute_cells(mesh);

        Profiler::EndTimer("VORONOI (par)");
    }

    // ============================================================
    // Cell construction (fast tier + slow tier + CPU fallback)
    // ============================================================

#ifndef CPU_DEBUG
    static hsize_t* d_face_offset             = nullptr;
    static int*     d_overflow_flag           = nullptr;
    static int*     d_fail_count              = nullptr;
    static int*     d_failed_indices          = nullptr;
    static int*     d_failed_count            = nullptr;
    static int      d_failed_indices_capacity = 0;
#endif

    void compute_cells(VMesh* mesh) {
        const int n_hydro = (int)mesh->n_hydro;

#ifndef CPU_DEBUG
        const int tpb    = _VORO_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;

        if (!d_face_offset) {
            d_face_offset   = gpu_calloc<hsize_t>(1);
            d_overflow_flag = gpu_calloc<int>(1);
            d_failed_count  = gpu_calloc<int>(1);
        }
        if (d_failed_indices_capacity < n_hydro) {
            if (d_failed_indices) gpu_free(d_failed_indices);
            d_failed_indices          = gpu_alloc<int>(n_hydro);
            d_failed_indices_capacity = n_hydro;
        }
        gpu_memset(d_face_offset, 0, sizeof(hsize_t));
        gpu_memset(d_overflow_flag, 0, sizeof(int));
        gpu_memset(d_failed_count, 0, sizeof(int));

        Profiler::StartGPU("kernel_compute_voronoi_cells_fast");
        kernel_compute_voronoi_cells_fast<<<blocks, tpb>>>(n_hydro,
                                                           (double*)mesh->knn->d_stored_points,
                                                           mesh->knn,
                                                           mesh->cell_status,
                                                           mesh,
                                                           d_face_offset,
                                                           d_overflow_flag);
        Profiler::EndGPU("kernel_compute_voronoi_cells_fast");

        {
            const int collect_tpb    = _MESH_BLOCK_SIZE_;
            const int collect_blocks = (n_hydro + collect_tpb - 1) / collect_tpb;
            Profiler::StartGPU("kernel_collect_failed_cells");
            kernel_collect_failed_cells<<<collect_blocks, collect_tpb>>>(
                n_hydro, mesh->cell_status, d_failed_indices, d_failed_count);
            Profiler::EndGPU("kernel_collect_failed_cells");
        }

        GPU_SYNC();
        const int n_failed        = *d_failed_count;
        const int n_hydro_global  = logging::sum_global((int)n_hydro);
        const int n_failed_global = logging::sum_global(n_failed);
        logging::root() << "VORONOI: Generated " << n_hydro_global << " cells. ("
                        << (100.0 * n_failed_global / (double)n_hydro_global) << "% slow tier)" << std::endl;

        if (n_failed > 0) {
            const int slow_blocks = (n_failed + tpb - 1) / tpb;
            Profiler::StartGPU("kernel_compute_voronoi_cells_slow");
            kernel_compute_voronoi_cells_slow<<<slow_blocks, tpb>>>(n_failed,
                                                                    d_failed_indices,
                                                                    (double*)mesh->knn->d_stored_points,
                                                                    mesh->knn,
                                                                    mesh->cell_status,
                                                                    mesh,
                                                                    d_face_offset,
                                                                    d_overflow_flag);
            Profiler::EndGPU("kernel_compute_voronoi_cells_slow");
        }

        GPU_SYNC();
        mesh->num_faces = *d_face_offset;
        if (*d_overflow_flag) {
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
#else
        unsigned long long face_offset   = 0;
        int                overflow_flag = 0;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int k = 0; k < n_hydro; k++) {
            if (overflow_flag) continue;
            const int seed_id = (int)mesh->real_sorted_ids[k];
            compute_single_voronoi_cell<_FAST_K_, _FAST_MAX_P_, _FAST_MAX_T_>(k,
                                                                              seed_id,
                                                                              (double*)mesh->knn->d_stored_points,
                                                                              mesh->knn,
                                                                              mesh->cell_status,
                                                                              mesh,
                                                                              &face_offset,
                                                                              &overflow_flag);
        }

        int n_failed = 0;
        for (int k = 0; k < n_hydro; k++)
            if (mesh->cell_status[k] != success) n_failed++;
        const int n_hydro_global  = logging::sum_global((int)n_hydro);
        const int n_failed_global = logging::sum_global(n_failed);
        logging::root() << "VORONOI: Generated " << n_hydro_global << " cells. ("
                        << (100.0 * n_failed_global / (double)n_hydro_global) << "% slow tier)" << std::endl;

        if (n_failed > 0) {
#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
            for (int k = 0; k < n_hydro; k++) {
                if (overflow_flag) continue;
                if (mesh->cell_status[k] == success) continue;
                const int seed_id = (int)mesh->real_sorted_ids[k];
                compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(k,
                                                                   seed_id,
                                                                   (double*)mesh->knn->d_stored_points,
                                                                   mesh->knn,
                                                                   mesh->cell_status,
                                                                   mesh,
                                                                   &face_offset,
                                                                   &overflow_flag);
            }
        }

        mesh->num_faces = (hsize_t)face_offset;
        if (overflow_flag) {
            std::cerr << "VORONOI: Error! face offset exceeds pre-allocated face capacity " << mesh->face_capacity
                      << ". Increase _FACE_CAPACITY_MULT_ in Config.sh." << std::endl;
            exit(EXIT_FAILURE);
        }
#endif

        // CPU fallback is invoked by compute_periodic_mesh after the halo-widening loop
        // converges, so compute_cells can be rerun with wider halos before perturbing seeds
    }

    // ============================================================
    // CUDA kernel wrappers
    // ============================================================
#ifndef CPU_DEBUG

    GLOBAL void kernel_init_cell_status(int n, Status* stat) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) stat[i] = security_radius_not_reached;
    }

    GLOBAL void kernel_count_failures(int n, const Status* stat, int* fail_count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n && stat[i] != success) atomicAdd(fail_count, 1);
    }

    GLOBAL void kernel_collect_failed_cells(int n, const Status* stat, int* failed_indices, int* failed_count) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n && stat[i] != success) {
            const int slot       = atomicAdd(failed_count, 1);
            failed_indices[slot] = i;
        }
    }

    GLOBAL __launch_bounds__(_VORO_BLOCK_SIZE_, 16) void kernel_compute_voronoi_cells_fast(int     n_hydro,
                                                                                           double* d_stored_points,
                                                                                           const knn_problem* knn,
                                                                                           Status*            stat,
                                                                                           VMesh*             mesh,
                                                                                           hsize_t* face_offset,
                                                                                           int*     overflow_flag) {
        const int k = blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        const int seed_id = (int)mesh->real_sorted_ids[k];
        compute_single_voronoi_cell<_FAST_K_, _FAST_MAX_P_, _FAST_MAX_T_>(
            k, seed_id, d_stored_points, knn, stat, mesh, (unsigned long long*)face_offset, overflow_flag);
    }

    GLOBAL __launch_bounds__(_VORO_BLOCK_SIZE_, 8) void kernel_compute_voronoi_cells_slow(int        n_failed,
                                                                                          const int* failed_ks,
                                                                                          double*    d_stored_points,
                                                                                          const knn_problem* knn,
                                                                                          Status*            stat,
                                                                                          VMesh*             mesh,
                                                                                          hsize_t* face_offset,
                                                                                          int*     overflow_flag) {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= n_failed) return;
        const int k       = failed_ks[i];
        const int seed_id = (int)mesh->real_sorted_ids[k];
        compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(
            k, seed_id, d_stored_points, knn, stat, mesh, (unsigned long long*)face_offset, overflow_flag);
    }

#endif // !CPU_DEBUG

    // ============================================================
    // Per-cell work (called by kernels and CPU loops)
    // ============================================================

    template <int K, int MAX_P, int MAX_T>
    HD void compute_single_voronoi_cell(int                 k,
                                        int                 seed_id,
                                        double*             d_stored_points,
                                        const knn_problem*  knn,
                                        Status*             stat,
                                        VMesh*              mesh,
                                        unsigned long long* face_offset,
                                        int*                overflow_flag) {

        unsigned int local_knn[K];
        knn::knn_for_point<K>(seed_id, knn, local_knn);

        BasicConvexCell<MAX_P, MAX_T> cell(seed_id, d_stored_points, &(stat[k]), mesh->buff);

        for (int v = 0; v < K; v++) {
            const unsigned int z = local_knn[v];
            cell.clip_by_plane(z);
            if (stat[k] != success) break;

            if (v >= 2 * DIMENSION &&
                cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * z))) {
                break;
            }
        }

        if (!cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * local_knn[K - 1]))) {
            stat[k] = security_radius_not_reached;
        }

        if (stat[k] == success) {
            const int     fc        = count_cell_faces(cell);
            const hsize_t my_offset = (hsize_t)portable_atomicAdd(face_offset, (unsigned long long)fc);
            if (my_offset + (hsize_t)fc > mesh->face_capacity) {
                portable_atomicExch(overflow_flag, 1);
                return;
            }
            mesh->face_counts[k] = (hsize_t)fc;
            mesh->face_ptr[k]    = my_offset;
            extract_cell_all(cell, mesh, (hsize_t)k);
        }
    }

    // ============================================================
    // CPU fallback for cells that failed both GPU tiers
    // ============================================================

    // permanently perturbed seed breaks symmetry -> so we have to rebuilt
    // not only the cell but also its neighbours. If two neighbours perturb -> cascade

    namespace {

        enum class FallbackOutcome { ok_unchanged, ok_perturbed, failed };

        struct CellSids {
            std::vector<int> offsets; // size n_hydro + 1
            std::vector<int> flat;    // size offsets.back()

            int        size_for(int k) const { return offsets[k + 1] - offsets[k]; }
            const int* begin_for(int k) const { return flat.data() + offsets[k]; }
        };

        CellSids build_cell_sids(const VMesh* mesh) {
            const int n_hydro = (int)mesh->n_hydro;
            const int n_seeds = (int)mesh->n_seeds;
            CellSids  cs;
            cs.offsets.assign(n_hydro + 1, 0);
            for (int sid = 0; sid < n_seeds; sid++) {
                const unsigned int k = mesh->sid_to_neighbor[sid];
                if ((int)k < n_hydro) cs.offsets[k + 1]++;
            }
            for (int k = 0; k < n_hydro; k++)
                cs.offsets[k + 1] += cs.offsets[k];
            cs.flat.resize(cs.offsets[n_hydro]);
            std::vector<int> cursor(n_hydro, 0);
            for (int sid = 0; sid < n_seeds; sid++) {
                const unsigned int k = mesh->sid_to_neighbor[sid];
                if ((int)k < n_hydro) cs.flat[cs.offsets[k] + cursor[k]++] = sid;
            }
            return cs;
        }

        void apply_perturbation(double*        d_stored_points,
                                int            seed_id,
                                int            attempt,
                                double         scale,
                                const int*     sids,
                                size_t         n_sids,
                                const double4* orig_positions) {
            unsigned int hash = (unsigned int)(seed_id * 2654435761u + attempt * 40503u);
            hash              = hash * 1103515245u + 12345u;
            const double dx   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
            hash              = hash * 1103515245u + 12345u;
            const double dy   = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
#ifdef dim_3D
            hash            = hash * 1103515245u + 12345u;
            const double dz = ((double)(hash & 0xFFFF) / 32768.0 - 1.0) * scale;
#endif
            for (size_t i = 0; i < n_sids; i++) {
                const int sid                        = sids[i];
                d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x + dx;
                d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y + dy;
#ifdef dim_3D
                d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z + dz;
#endif
            }
        }

        void
        rewind_perturbation(double* d_stored_points, const int* sids, size_t n_sids, const double4* orig_positions) {
            for (size_t i = 0; i < n_sids; i++) {
                const int sid                        = sids[i];
                d_stored_points[DIMENSION * sid + 0] = orig_positions[i].x;
                d_stored_points[DIMENSION * sid + 1] = orig_positions[i].y;
#ifdef dim_3D
                d_stored_points[DIMENSION * sid + 2] = orig_positions[i].z;
#endif
            }
        }

        void append_cell_to_mesh(VMesh* mesh, int k, const ConvexCell& cell) {
            const int fc = count_cell_faces(cell);
            ensure_face_capacity(mesh, mesh->num_faces + fc);
            mesh->face_ptr[k]    = mesh->num_faces;
            mesh->face_counts[k] = (hsize_t)fc;
            extract_cell_all(cell, mesh, (hsize_t)k);
            mesh->num_faces += (hsize_t)fc;
        }

        FallbackOutcome
        rebuild_cell_with_perturb_retry(VMesh* mesh, int k, double* d_stored_points, const CellSids& cell_sids) {
            const int     seed_id  = (int)mesh->real_sorted_ids[k];
            const double4 seed_pos = point_from_ptr(d_stored_points + DIMENSION * seed_id);
            const int*    sids     = cell_sids.begin_for(k);
            const size_t  n_sids   = (size_t)cell_sids.size_for(k);

            std::vector<double4> orig_positions(n_sids);
            for (size_t i = 0; i < n_sids; i++)
                orig_positions[i] = point_from_ptr(d_stored_points + DIMENSION * sids[i]);

            // exhaustive distance-sorted clip list
            const int                           n_seeds = (int)mesh->n_seeds;
            std::vector<std::pair<double, int>> dists;
            dists.reserve(n_seeds - 1);
            for (int j = 0; j < n_seeds; j++) {
                if (j == seed_id) continue;
                const double4 other = point_from_ptr(d_stored_points + DIMENSION * j);
                const double  dx    = other.x - seed_pos.x;
                const double  dy    = other.y - seed_pos.y;
                const double  dz    = other.z - seed_pos.z;
                dists.push_back({dx * dx + dy * dy + dz * dz, j});
            }
            std::sort(dists.begin(), dists.end());

            constexpr int max_perturb = 9;
            double        scale       = 1e-13;

            for (int attempt = 0; attempt <= max_perturb; attempt++) {
                if (attempt > 0)
                    apply_perturbation(d_stored_points, seed_id, attempt, scale, sids, n_sids, orig_positions.data());

                Status     status = success;
                ConvexCell cell(seed_id, d_stored_points, &status, mesh->buff);
                for (size_t di = 0; di < dists.size(); di++) {
                    const int j = dists[di].second;
                    cell.clip_by_plane(j);
                    if (cell.is_security_radius_reached(point_from_ptr(d_stored_points + DIMENSION * j))) break;
                    if (status != success) break;
                }

                if (status == success) {
                    append_cell_to_mesh(mesh, k, cell);
                    mesh->cell_status[k] = success;
                    std::cout << "VORONOI: cell " << k << " fallback succeeded (attempt " << attempt << ")."
                              << std::endl;
                    return (attempt == 0) ? FallbackOutcome::ok_unchanged : FallbackOutcome::ok_perturbed;
                }

                if (attempt > 0) rewind_perturbation(d_stored_points, sids, n_sids, orig_positions.data());
                scale *= 10.0;
            }
            return FallbackOutcome::failed;
        }

        std::vector<int> collect_unique_neighbors(const VMesh* mesh, const std::vector<int>& sources) {
            const int         n_hydro = (int)mesh->n_hydro;
            std::vector<bool> seen(n_hydro, false);
            std::vector<int>  result;
            for (int k : sources) {
                const hsize_t fp = mesh->face_ptr[k];
                const hsize_t fc = mesh->face_counts[k];
                for (hsize_t f = 0; f < fc; f++) {
                    const int kn = mesh->neighbor_cell[fp + f];
                    if (kn < 0 || kn >= n_hydro) continue; // box-boundary plane
                    if (seen[kn]) continue;
                    seen[kn] = true;
                    result.push_back(kn);
                }
            }
            return result;
        }

        // initial_perturbed cells stay eligible for round-1 rebuild: if K_a, K_b both perturb
        // AND are mutual neighbours, K_a was built against K_b_original and needs the rebuild.
        void run_symmetry_pass(VMesh*                  mesh,
                               double*                 d_stored_points,
                               const CellSids&         cell_sids,
                               const std::vector<int>& initial_perturbed) {
            constexpr int MAX_ROUNDS = 4;

            std::vector<int>   work        = initial_perturbed;
            int                rebuilt     = 0;
            int                rounds      = 0;
            unsigned long long face_offset = (unsigned long long)mesh->num_faces;
            int                overflow    = 0;

            while (!work.empty()) {
                if (++rounds > MAX_ROUNDS) {
                    std::cerr << "VORONOI: symmetry cascade did not converge after " << MAX_ROUNDS
                              << " rounds, aborting." << std::endl;
                    exit(EXIT_FAILURE);
                }

                std::vector<int> affected = collect_unique_neighbors(mesh, work);
                if (affected.empty()) break;

                std::vector<int> next_work;
                for (int kn : affected) {
                    mesh->cell_status[kn] = security_radius_not_reached;
                    const int seed_id     = (int)mesh->real_sorted_ids[kn];
                    compute_single_voronoi_cell<_K_, _MAX_P_, _MAX_T_>(
                        kn, seed_id, d_stored_points, mesh->knn, mesh->cell_status, mesh, &face_offset, &overflow);
                    if (overflow) {
                        std::cerr << "VORONOI: face overflow during symmetry rebuild — "
                                     "increase _FACE_CAPACITY_MULT_ in Config.sh."
                                  << std::endl;
                        exit(EXIT_FAILURE);
                    }

                    if (mesh->cell_status[kn] != success) {
                        // KNN-based slow tier hit a degeneracy; fall through to perturb path.
                        // Sync mesh->num_faces with face_offset because the helper appends via num_faces.
                        mesh->num_faces         = (hsize_t)face_offset;
                        FallbackOutcome outcome = rebuild_cell_with_perturb_retry(mesh, kn, d_stored_points, cell_sids);
                        if (outcome == FallbackOutcome::failed) {
                            std::cerr << "VORONOI: symmetry rebuild for cell " << kn << " all fallback attempts FAILED."
                                      << std::endl;
#ifdef USE_MPI
                            MPI_Abort(MPI_COMM_WORLD, 1);
#else
                            exit(EXIT_FAILURE);
#endif
                        }
                        face_offset = (unsigned long long)mesh->num_faces;
                        if (outcome == FallbackOutcome::ok_perturbed) next_work.push_back(kn);
                    }
                    rebuilt++;
                }
                work = std::move(next_work);
            }
            mesh->num_faces = (hsize_t)face_offset;

            std::cout << "VORONOI: " << initial_perturbed.size() << " cell(s) permanently perturbed; " << rebuilt
                      << " neighbour rebuild(s) over " << rounds << " round(s)." << std::endl;
        }

        void compact_face_arrays(VMesh* mesh) {
            const hsize_t cap     = mesh->num_faces;
            const int     n_hydro = (int)mesh->n_hydro;

            std::vector<int>    neighbor_tmp(cap);
            std::vector<double> area_tmp(cap);
#ifdef MOVING_MESH
            std::vector<double> fmid_tmp(cap * (DIMENSION - 1));
#endif

            hsize_t out = 0;
            for (int k = 0; k < n_hydro; k++) {
                const hsize_t fp = mesh->face_ptr[k];
                const hsize_t fc = mesh->face_counts[k];
                for (hsize_t i = 0; i < fc; i++) {
                    neighbor_tmp[out + i] = mesh->neighbor_cell[fp + i];
                    area_tmp[out + i]     = mesh->face_area[fp + i];
                }
#ifdef MOVING_MESH
                for (hsize_t i = 0; i < fc * (DIMENSION - 1); i++)
                    fmid_tmp[out * (DIMENSION - 1) + i] = mesh->f_mid_local[fp * (DIMENSION - 1) + i];
#endif
                mesh->face_ptr[k] = out;
                out += fc;
            }

            for (hsize_t i = 0; i < out; i++) {
                mesh->neighbor_cell[i] = neighbor_tmp[i];
                mesh->face_area[i]     = area_tmp[i];
            }
#ifdef MOVING_MESH
            for (hsize_t i = 0; i < out * (DIMENSION - 1); i++)
                mesh->f_mid_local[i] = fmid_tmp[i];
#endif
            mesh->num_faces = out;
        }

    } // namespace

    int cpu_fallback_failed_cells(VMesh* mesh) {
        const int n_hydro         = (int)mesh->n_hydro;
        Status*   stat            = mesh->cell_status;
        double*   d_stored_points = (double*)mesh->knn->d_stored_points;

        int num_failed = 0;
#ifndef CPU_DEBUG
        if (!d_fail_count) d_fail_count = gpu_calloc<int>(1);
        gpu_memset(d_fail_count, 0, sizeof(int));
        {
            const int tpb    = _MESH_BLOCK_SIZE_;
            const int blocks = (n_hydro + tpb - 1) / tpb;
            kernel_count_failures<<<blocks, tpb>>>(n_hydro, stat, d_fail_count);
        }
        GPU_SYNC();
        num_failed = *d_fail_count;
        if (num_failed == 0) return 0;
        gpu_prefetch_to_cpu(stat, n_hydro * sizeof(Status));
#else
        for (int k = 0; k < n_hydro; k++)
            if (stat[k] != success) num_failed++;
        if (num_failed == 0) return 0;
#endif

        std::cout << "VORONOI: " << num_failed << " cells failed, retrying with fallback..." << std::endl;

        const CellSids   cell_sids = build_cell_sids(mesh);
        std::vector<int> perturbed_ks;

        for (int k = 0; k < n_hydro; k++) {
            if (stat[k] == success) continue;

            const Status original = stat[k];
            if (original != security_radius_not_reached && original != needs_exact_predicates) {
                std::cerr << "VORONOI: cell " << k << " failed with unrecoverable status: " << original << std::endl;
#ifdef USE_MPI
                MPI_Abort(MPI_COMM_WORLD, 1);
#else
                exit(EXIT_FAILURE);
#endif
            }
            std::cout << "VORONOI: cell " << k << " failed with status: " << original << std::endl;

            switch (rebuild_cell_with_perturb_retry(mesh, k, d_stored_points, cell_sids)) {
            case FallbackOutcome::ok_unchanged:
                break;
            case FallbackOutcome::ok_perturbed:
                perturbed_ks.push_back(k);
                break;
            case FallbackOutcome::failed:
                std::cerr << "VORONOI: cell " << k << " all fallback attempts FAILED, aborting." << std::endl;
#ifdef USE_MPI
                MPI_Abort(MPI_COMM_WORLD, 1);
#else
                exit(EXIT_FAILURE);
#endif
            }
        }

        if (!perturbed_ks.empty()) {
            run_symmetry_pass(mesh, d_stored_points, cell_sids, perturbed_ks);
            compact_face_arrays(mesh);
        }
        return (int)perturbed_ks.size();
    }

    // ============================================================
    // Halo completeness sentinel
    // ============================================================

#ifndef CPU_DEBUG
    GLOBAL void kernel_halo_completeness_check(hsize_t              n_hydro,
                                               const knn_problem*   knn,
                                               const unsigned int*  real_sorted_ids,
                                               int                  pts_mpi_base,
                                               int                  n_mpi_ghosts,
                                               const unsigned char* is_outer_layer,
                                               int*                 d_flag) {
        const hsize_t k = (hsize_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (k >= n_hydro) return;
        if (*d_flag) return;

        const int    sid_self = (int)real_sorted_ids[k];
        unsigned int knearest[_K_];
        knn::knn_for_point<_K_>(sid_self, knn, knearest);

        for (int i = 0; i < _K_; i++) {
            const unsigned int sid  = knearest[i];
            const unsigned int orig = knn->d_permutation[sid];
            if ((int)orig < pts_mpi_base) continue;
            const int slot = (int)orig - pts_mpi_base;
            if (slot < 0 || slot >= n_mpi_ghosts) continue;
            if (is_outer_layer[slot]) {
                atomicOr(d_flag, 1);
                return;
            }
        }
    }
#endif

    int halo_completeness_flag(VMesh* mesh, int n_pgh) {
        if (proteus_mpi::g_halo.n_neighbors == 0 || proteus_mpi::g_halo.n_mpi_ghosts == 0) return 0;

        const int n_hydro      = (int)mesh->n_hydro;
        const int pts_mpi_base = n_hydro + n_pgh;
        const int n_mpi        = proteus_mpi::g_halo.n_mpi_ghosts;

#ifndef CPU_DEBUG
        static int* d_flag = nullptr;
        if (!d_flag) d_flag = gpu_alloc<int>(1);
        *d_flag = 0;

        const int tpb    = _MESH_BLOCK_SIZE_;
        const int blocks = (n_hydro + tpb - 1) / tpb;
        kernel_halo_completeness_check<<<blocks, tpb>>>(
            n_hydro, mesh->knn, mesh->real_sorted_ids, pts_mpi_base, n_mpi, proteus_mpi::g_halo.is_outer_layer, d_flag);
        GPU_LAUNCH_CHECK();
        GPU_SYNC();
        return *d_flag;
#else
        unsigned int knearest[_K_];
        for (int k = 0; k < n_hydro; k++) {
            const int sid_self = (int)mesh->real_sorted_ids[k];
            knn::knn_for_point<_K_>(sid_self, mesh->knn, knearest);
            for (int i = 0; i < _K_; i++) {
                const unsigned int orig = mesh->knn->d_permutation[knearest[i]];
                if ((int)orig < pts_mpi_base) continue;
                const int slot = (int)orig - pts_mpi_base;
                if (slot < 0 || slot >= n_mpi) continue;
                if (proteus_mpi::g_halo.is_outer_layer[slot]) return 1;
            }
        }
        return 0;
#endif
    }

} // namespace voronoi
