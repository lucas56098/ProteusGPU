# Configuration file for compilation options
# Makefile converts these to -D flags

################################################################
# setup
################################################################

#dim_2D                             # run in 2D mode
dim_3D                              # run in 3D mode

#CUDA                               # run in GPU mode
CPU_DEBUG                           # run in CPU mode

#OUTPUT_MESH                        # dump full Voronoi geometry (CSR faces + volumes) in snapshots

#ENABLE_PROFILING                   # hierarchical timers and profile.hdf5
#CUDA_PROFILING                     # profiling of GPU kernels (needs ENABLE_PROFILING)

################################################################
# parallelization
################################################################

USE_OPENMP                          # enable multithreading on CPU
#USE_MPI                            # enable MPI (1 GPU per rank when CUDA is enabled)
#GPU_AWARE_MPI                      # requires CUDA + USE_MPI + CUDA-aware MPI lib

################################################################
# hydro
################################################################

MOVING_MESH                         # enable moving mesh hydrodynamics
#_CELL_SHAPING_SPEED_=0.7           # mesh regularization speed
#_CELL_SHAPING_FACTOR_=0.2          # regularization threshold in cell radii
#VOL_REGULARIZE=1                   # volume regularization: threshold in Ri_ref/Ri
#_VOL_SHAPING_SPEED_=0.7            # drift speed as a fraction of max(c_s, |v_gas|)

#_GAMMA_EOS_=1.6666666666666667     # adiabatic index

################################################################
# astrophysics source terms
################################################################

#ASTRO_PHYSICS                       # main switch

#NFW                                 # static NFW potential
#HERNQUIST                           # static Hernquist potential
#SMBH                                # softened central point mass potential

#COOLING                             # radiative cooling (Townsend 2009)

#SF_FEEDBACK                         # SNIa injection + particle-free SF heating

#AGN_THERMAL                         # AGN thermal mode
#AGN_KINETIC                         # AGN kinetic mode
#LIMITERS                            # central-region clamps on T and |v|

################################################################
# cell construction array sizes
################################################################
# K is the max number of KNN to be checked
# P and T are the max clipping planes / triangles stored per cell

# fast-tier (overflowing cells -> slow tier)
#_FAST_K_=35                         # default 2D/3D ~15/35
#_FAST_MAX_P_=30                     # default 2D/3D ~20/30
#_FAST_MAX_T_=60                     # default 2D/3D ~20/60

# slow tier (overflowing cells -> CPU-fallback)
#_K_=190                             # default 2D/3D ~35/190
#_MAX_P_=50                          # default 2D/3D ~30/50
#_MAX_T_=96                          # default 2D/3D ~60/96

# CPU-fallback using 32-bit indices
#_BIG_MAX_P_=1024                    # default 1024
#_BIG_MAX_T_=2048                    # default 2048

#_FACE_CAPACITY_MULT_=17             # faces allocated per cell (default 2D/3D ~8/17)

################################################################
# GPU Kernel block sizes
################################################################

# GPU kernel block sizes
#_VORO_BLOCK_SIZE_=64               # voronoi cell computation
#_KNN_BLOCK_SIZE_=256               # KNN grid sort
#_GRAD_BLOCK_SIZE_=256              # gradient computation
#_HYDRO_BLOCK_SIZE_=256             # hydro kernels
#_MESH_BLOCK_SIZE_=256              # periodic mesh / ghost / scaling
#_MPI_PACK_BLOCK_SIZE_=256          # kernels preparing the MPI comm
