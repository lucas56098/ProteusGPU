# Configuration file for compilation options
# Makefile converts these to -D flags

################################################################
# setup
################################################################

#dim_2D                             # run in 2D mode
dim_3D                              # run in 3D mode

#CUDA                                # run in GPU mode
CPU_DEBUG                          # run in CPU mode

#ENABLE_PROFILING                    # enable the hierarchical timers and profile.hdf5 (off = compiled away)
#CUDA_PROFILING                      # enable profiling of GPU kernels (needs ENABLE_PROFILING)

################################################################
# hydro
################################################################

MOVING_MESH                         # enable moving mesh hydrodynamics
#_CELL_SHAPING_SPEED_=0.7            # mesh regularization speed fraction (default 0.7)
#_CELL_SHAPING_FACTOR_=0.2           # regularization threshold in cell radii (default 0.2)

#_GAMMA_EOS_=1.6666666666666667      # adiabatic index (default 5/3)

################################################################
# parallelization
################################################################

USE_OPENMP                          # enable multithreading on CPU (uses all available cores)
#USE_MPI                             # enable MPI / multi-node parallelization (1 GPU per rank when CUDA is enabled)
#GPU_AWARE_MPI                       # pass device pointers directly to MPI (requires CUDA + USE_MPI + CUDA-aware MPI lib)

# GPU kernel block sizes
#_VORO_BLOCK_SIZE_=64                # voronoi cell computation (register-heavy, default 64)
#_KNN_BLOCK_SIZE_=256                # KNN grid sort kernels (default 256)
#_GRAD_BLOCK_SIZE_=256               # gradient computation kernel (default 256)
#_HYDRO_BLOCK_SIZE_=256              # hydro flux / CFL / copy / volume kernels (default 256)
#_MESH_BLOCK_SIZE_=256               # periodic mesh / ghost / scaling kernels (default 256)
#_MPI_PACK_BLOCK_SIZE_=256           # kernels preparing the MPI comm

################################################################
# compile time memory constraints
################################################################

#_K_=190                             # KNN candidates, slow tier            (default 2D/3D ~35/190)
#_MAX_P_=50                          # max clipping planes per cell, slow   (default 2D/3D ~30/50)
#_MAX_T_=96                          # max triangles per cell, slow         (default 2D/3D ~60/96)

# fast-tier limits: cells that overflow these fall back to slow tier (above)
#_FAST_K_=35                         # KNN candidates, fast tier            (default 2D/3D ~15/35)
#_FAST_MAX_P_=30                     # max clipping planes per cell, fast   (default 2D/3D ~20/30)
#_FAST_MAX_T_=60                     # max triangles per cell, fast         (default 2D/3D ~20/60)

#_FACE_CAPACITY_MULT_=17             # max face array entries allocated per cell (default 2D/3D ~8/17)

# wide CPU-fallback tier: 32-bit indices, used when a cell overflows the 8-bit limits above
#_BIG_MAX_P_=1024                    # max clipping planes per cell, wide tier (default 1024)
#_BIG_MAX_T_=2048                    # max triangles per cell, wide tier (default 2048)
