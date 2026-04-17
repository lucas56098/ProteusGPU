# Configuration file for compilation options
# Makefile converts these to -D flags

################################################################
# setup
################################################################

#dim_2D                             # run in 2D mode
dim_3D                              # run in 3D mode

#CUDA                               # run in GPU mode
CPU_DEBUG                           # run in CPU mode

USE_HDF5                            # HDF5 for IC and output (mandatory)

################################################################
# hydro
################################################################

MOVING_MESH                         # enable moving mesh hydrodynamics
_CELL_SHAPING_SPEED_=0.7            # mesh regularization speed fraction
_CELL_SHAPING_FACTOR_=0.2           # regularization threshold in cell radii

_GAMMA_EOS_=1.6666666666666667      # adiabatic index

#RIEMANN_HLL                        # use HLL riemann solver
RIEMANN_HLLC                        # use HLLC riemann solver

################################################################
# parallelization
################################################################

#USE_OPENMP                          # enable multithreading on CPU
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

_K_=190                             # number of nearest neighbors (KNN)
_MAX_P_=50                          # max number of clipping planes per Voronoi cell
_MAX_T_=96                          # max number of triangles per Voronoi cell
_FACE_CAPACITY_MULT_=17             # max face array entries allocated per cell

################################################################
# experimental / debug
################################################################

#DEBUG_MODE                         # verbose printout
#ENABLE_PROFILING                   # profiling of main routines
#SAVE_MEMORY                        # float instead of double for selected variables
