#pragma once

// The ifdef switches of Config.sh as constexpr flags.

#if defined(dim_2D) == defined(dim_3D)
#error "CONFIG: switch on exactly one of dim_2D and dim_3D"
#endif

#if defined(GPU_AWARE_MPI) && !(defined(CUDA) && defined(USE_MPI))
#error "CONFIG: GPU_AWARE_MPI needs CUDA and USE_MPI"
#endif

#if (defined(REGULARIZE_MESH_DISTANCE) || defined(REGULARIZE_MESH_FACE_ANGLE) || defined(REGULARIZE_MESH_VOLUME)) &&   \
    !defined(MOVING_MESH)
#error "CONFIG: REGULARIZE_MESH_DISTANCE, REGULARIZE_MESH_FACE_ANGLE and REGULARIZE_MESH_VOLUME need MOVING_MESH"
#endif

#if defined(CUDA_PROFILING) && !(defined(ENABLE_PROFILING) && defined(CUDA))
#error "CONFIG: CUDA_PROFILING needs ENABLE_PROFILING and CUDA"
#endif

#if !defined(ASTRO_PHYSICS) &&                                                                                         \
    (defined(NFW) || defined(HERNQUIST) || defined(SMBH) || defined(COOLING) || defined(SF_FEEDBACK) ||                \
     defined(AGN_THERMAL) || defined(AGN_KINETIC) || defined(LIMITERS))
#error "CONFIG: the astrophysical sources need ASTRO_PHYSICS"
#endif

// HD marks a function that can be called on the CPU and GPU
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

namespace config {

#ifdef dim_2D
    constexpr int dim = 2;
#else
    constexpr int dim = 3;
#endif

#ifdef CUDA
    constexpr bool use_gpu = true;
#else
    constexpr bool use_gpu = false;
#endif

#ifdef __CUDA_ARCH__ // tells HD functions if compiled on GPU
    constexpr bool compiling_for_gpu = true;
#else
    constexpr bool compiling_for_gpu = false;
#endif

#ifdef USE_MPI
    constexpr bool use_mpi = true;
#else
    constexpr bool use_mpi = false;
#endif

#ifdef USE_OPENMP
    constexpr bool use_openmp = true;
#else
    constexpr bool use_openmp = false;
#endif

#ifdef MOVING_MESH
    constexpr bool moving_mesh = true;
#else
    constexpr bool moving_mesh = false;
#endif

#ifdef INDIVIDUAL_TIMESTEPS
    constexpr bool individual_timesteps = true;
#else
    constexpr bool individual_timesteps = false;
#endif

#ifdef OUTPUT_DEBUG
    constexpr bool output_debug = true;
#else
    constexpr bool output_debug = false;
#endif

#ifdef ENABLE_PROFILING
    constexpr bool profiling = true;
#else
    constexpr bool profiling = false;
#endif
#ifdef CUDA_PROFILING
    constexpr bool cuda_profiling = true;
#else
    constexpr bool cuda_profiling = false;
#endif

// astrophysical sources: potentials, SMBH, cooling, feedback, limiters
#ifdef ASTRO_PHYSICS
    constexpr bool astro_physics = true;
#else
    constexpr bool astro_physics = false;
#endif
#ifdef NFW
    constexpr bool nfw = true;
#else
    constexpr bool nfw = false;
#endif
#ifdef HERNQUIST
    constexpr bool hernquist = true;
#else
    constexpr bool hernquist = false;
#endif
#ifdef SMBH
    constexpr bool smbh = true;
#else
    constexpr bool smbh = false;
#endif
#ifdef COOLING
    constexpr bool cooling = true;
#else
    constexpr bool cooling = false;
#endif
#ifdef SF_FEEDBACK
    constexpr bool star_feedback = true;
#else
    constexpr bool star_feedback = false;
#endif
#ifdef AGN_THERMAL
    constexpr bool agn_thermal = true;
#else
    constexpr bool agn_thermal = false;
#endif
#ifdef AGN_KINETIC
    constexpr bool agn_kinetic = true;
#else
    constexpr bool agn_kinetic = false;
#endif
#ifdef LIMITERS
    constexpr bool limiters = true;
#else
    constexpr bool limiters = false;
#endif
    constexpr bool static_potential = nfw || hernquist || smbh;
    constexpr bool agn              = agn_thermal || agn_kinetic;

// moving mesh regularization (keeping the cells "round")
#ifdef REGULARIZE_MESH_DISTANCE
    constexpr double regularization_speed_by_distance = REGULARIZE_MESH_DISTANCE;
#else
    constexpr double regularization_speed_by_distance = 0.7;
#endif
#ifdef REGULARIZE_MESH_FACE_ANGLE
    constexpr double regularization_speed_by_face_angle = REGULARIZE_MESH_FACE_ANGLE;
#else
    constexpr double regularization_speed_by_face_angle = 0.0;
#endif
#ifdef REGULARIZE_MESH_VOLUME
    constexpr double regularization_speed_by_volume = REGULARIZE_MESH_VOLUME;
#else
    constexpr double regularization_speed_by_volume = 0.0;
#endif

// mesh building
#ifdef TREE_LEAF_POINTS
    constexpr int tree_leaf_points = TREE_LEAF_POINTS;
#else
    constexpr int tree_leaf_points = 8;
#endif
#ifdef MESH_MIN_CELLS_ON_GPU
    constexpr int mesh_min_cells_on_gpu = MESH_MIN_CELLS_ON_GPU;
#else
    constexpr int mesh_min_cells_on_gpu = 1000;
#endif

#ifdef ADIABATIC_INDEX
    constexpr double adiabatic_index = ADIABATIC_INDEX;
#else
    constexpr double adiabatic_index = 5.0 / 3.0; // monoatomic gas
#endif

} // namespace config
