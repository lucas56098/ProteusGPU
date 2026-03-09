# Configuration file for compilation options
# Makefile converts these to -D flags

# 2D or 3D version of the code (one must be defined)
#dim_2D
dim_3D

# GPU settings
#CUDA -- not yet implemented
#HIP -- not yet implemented
CPU_DEBUG # -- mandatory for now

# HDF5 for IC and output (currently mandatory)
USE_HDF5

# Debug
#DEBUG_MODE # also enables writing voronoi edges...
#ENABLE_PROFILING # timing option

# enable OpenMP parallelization for CPU mode (requires g++-15 on macOS)
USE_OPENMP

# Riemann solver
#RIEMANN_HLL
RIEMANN_HLLC


# Output types
#WRITE_KNN_OUTPUT

# Verification (bruteforce KNN check)
#VERIFY

# Compile-time constants for KNN and Voronoi
_K_=190                  # number of nearest neighbors
_MAX_P_=64              # max number of clipping planes per Voronoi cell
_MAX_T_=96              # max number of triangles per Voronoi cell

_KNN_BLOCK_SIZE_=32     # number of threads per block for KNN
_VORO_BLOCK_SIZE_=16    # number of threads per block for Voronoi
_OMP_HYDRO_THREADS_=16 # number of threads for hydro solver