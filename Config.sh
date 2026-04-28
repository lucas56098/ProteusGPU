# Configuration file for compilation options
# Makefile converts these to -D flags

################################################################
# setup
################################################################

#dim_2D                             # run in 2D mode
dim_3D                              # run in 3D mode

#CUDA                               # run in GPU mode
CPU_DEBUG                           # run in CPU mode

#ENABLE_PROFILING                   # profiling of main routines

################################################################
# hydro
################################################################

MOVING_MESH                         # enable moving mesh hydrodynamics
_CELL_SHAPING_SPEED_=0.7            # mesh regularization speed fraction
_CELL_SHAPING_FACTOR_=0.2           # regularization threshold in cell radii

_GAMMA_EOS_=1.6666666666666667      # adiabatic index

################################################################
# parallelization
################################################################

USE_OPENMP                          # enable multithreading on CPU
_OMP_HYDRO_THREADS_=16              # threads for hydro solver

# GPU kernel block sizes
_VORO_BLOCK_SIZE_=64                # voronoi cell computation (register-heavy)
_KNN_BLOCK_SIZE_=256                # KNN grid sort kernels
_GRAD_BLOCK_SIZE_=256               # gradient computation kernel
_HYDRO_BLOCK_SIZE_=256              # hydro flux / CFL / copy / volume kernels
_MESH_BLOCK_SIZE_=256               # periodic mesh / ghost / scaling kernels

################################################################
# compile time memory constraints
################################################################

_K_=190                             # KNN candidates, slow tier            (2D/3D ~35/190)
_MAX_P_=50                          # max clipping planes per cell, slow   (2D/3D ~30/50)
_MAX_T_=96                          # max triangles per cell, slow         (2D/3D ~60/96)

# fast-tier limits: cells that overflow these fall back to slow tier (above)
_FAST_K_=35                         # KNN candidates, fast tier            (2D/3D ~15/35)
_FAST_MAX_P_=30                     # max clipping planes per cell, fast   (2D/3D ~20/30)
_FAST_MAX_T_=60                     # max triangles per cell, fast         (2D/3D ~20/60)

_FACE_CAPACITY_MULT_=17             # max face array entries allocated per cell (2D/3D ~8/17)
