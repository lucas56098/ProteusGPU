# parallel build is default
MAKEFLAGS += -j

# set config/build_dir/exec paths
CONFIG ?= Config.sh
BUILD_DIR ?= build
EXEC ?= ProteusGPU

# load Config.sh options and convert them to -D flags
SHELL := /bin/bash
CONFIG_DEFINES := $(shell grep -v "^\#" $(CONFIG) | grep -v "^$$" | grep -v "^!" | awk 'NF {print "-D" $$1}')
DEFINES :=

# set systype
ifdef SYSTYPE
        SYSTYPE := $(SYSTYPE)
else
        SYSTYPE ?= $(shell uname -s)
        -include Makefile.systype
endif

# check CUDA / PROFILING / MPI
CUDA_ENABLED := $(findstring CUDA,$(CONFIG_DEFINES))
PROFILING_ENABLED := $(findstring ENABLE_PROFILING,$(CONFIG_DEFINES))
MPI_ENABLED := $(findstring USE_MPI,$(CONFIG_DEFINES))

# ============================================================
# CUDA mode: nvcc compiler flags
# ============================================================
ifeq ($(CUDA_ENABLED),CUDA)

CXXFLAGS = --compiler-options -Wall,-Wextra,-Wno-unknown-pragmas -std=c++14
CXXFLAGS += --expt-relaxed-constexpr
CXXFLAGS += -dc -O3 --prec-div=false --prec-sqrt=false --ftz=true --fmad=true
LDFLAGS =
BUILD_MODE_MESSAGE = CUDA RELEASE

# optionally enable openmp
ifneq (,$(findstring USE_OPENMP,$(CONFIG_DEFINES)))
	CXXFLAGS += --compiler-options -fopenmp
	LDFLAGS += --compiler-options -fopenmp
	OPENMP_MESSAGE = OpenMP enabled
else
	OPENMP_MESSAGE = OpenMP disabled
endif

CUDA_MESSAGE = CUDA enabled


# ============================================================
# CPU_DEBUG mode: host compiler flags (g++/clang++)
# ============================================================
else

CXXFLAGS = -Wall -Wextra -std=c++14 -O3
LDFLAGS =
BUILD_MODE_MESSAGE = RELEASE

# g++ does not recognize .cu extension — treat as C++
CXXFLAGS += -x c++

# optionally enable openmp
ifneq (,$(findstring USE_OPENMP,$(CONFIG_DEFINES)))
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
	OPENMP_MESSAGE = OpenMP enabled
else
	OPENMP_MESSAGE = OpenMP disabled
endif

CUDA_MESSAGE = CUDA disabled (CPU_DEBUG)

endif # CUDA_ENABLED

CXXFLAGS += -MMD -MP
INCLUDES = -Isrc -Isrc/global

# ============================================================
# Directories and object files
# ============================================================
SRC_DIR = src
GLOBAL_DIR = $(SRC_DIR)/global
IO_DIR = $(SRC_DIR)/io
KNN_DIR = $(SRC_DIR)/knn
BEGRUN_DIR = $(SRC_DIR)/begrun
VORONOI_DIR = $(SRC_DIR)/voronoi
HYDRO_DIR = $(SRC_DIR)/hydro
GRADIENTS_DIR = $(SRC_DIR)/gradients
PROFILER_DIR = $(SRC_DIR)/profiler
MPI_DIR = $(SRC_DIR)/mpi

MAIN_OBJ = $(BUILD_DIR)/main.o
GLOBAL_OBJ = $(BUILD_DIR)/globals.o $(BUILD_DIR)/log.o
IO_OBJ = $(BUILD_DIR)/input.o $(BUILD_DIR)/output.o
KNN_OBJ = $(BUILD_DIR)/knn.o
BEGRUN_OBJ = $(BUILD_DIR)/begrun.o
VORONOI_OBJ = $(BUILD_DIR)/voronoi.o $(BUILD_DIR)/moving.o
HYDRO_OBJ = $(BUILD_DIR)/finite_volume_solver.o
GRADIENTS_OBJ = $(BUILD_DIR)/gradients.o
PROFILER_OBJ = $(BUILD_DIR)/profiler.o
MPI_OBJ = $(BUILD_DIR)/mpi_compat.o $(BUILD_DIR)/decomp.o $(BUILD_DIR)/halo.o $(BUILD_DIR)/migrate.o
OBJECTS = $(MAIN_OBJ) $(GLOBAL_OBJ) $(IO_OBJ) $(KNN_OBJ) $(BEGRUN_OBJ) $(VORONOI_OBJ) $(HYDRO_OBJ) $(GRADIENTS_OBJ) $(PROFILER_OBJ) $(MPI_OBJ)

# name of executable
TARGET = $(EXEC)

# ============================================================
# Systypes
# ============================================================
ifeq ($(SYSTYPE),Ubuntu)
# requires hdf5 installed
	CXX_RELEASE = g++
	HDF5_CFLAGS ?= -I/usr/include/hdf5/serial
	HDF5_LIBS ?= -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
	CUDA_ARCH ?= sm_89
endif

ifeq ($(SYSTYPE),macOS)
# requires hdf5 installed via homebrew and g++-15 as compiler (apple clang does not support openmp...)
# also only CPU_DEBUG works here of course
	CXX_RELEASE = g++-15
	HDF5_CFLAGS ?= -I/opt/homebrew/opt/hdf5/include
	HDF5_LIBS ?= -L/opt/homebrew/opt/hdf5/lib -lhdf5
endif

ifeq ($(SYSTYPE),MPCDF)
# VERA (A100): module load gcc/15 hdf5-serial/1.12.2 cuda/13.0 
# BinAC2 (A100): module load compiler/gnu/14.2 lib/hdf5/1.12-gnu-14.2 devel/cuda/13.0
	CXX_RELEASE = g++
    HDF5_CFLAGS ?= -I${HDF5_HOME}/include
    HDF5_LIBS ?= -L${HDF5_HOME}/lib -lhdf5
	CUDA_ARCH ?= sm_80
endif

ifeq ($(SYSTYPE),Jupiter)
# JUWELS Booster (A100): module load CUDA/13 HDF5/1.14.6-serial
        CXX_RELEASE = g++
        HDF5_CFLAGS ?= -I/p/software/default/stages/2026/software/HDF5/1.14.6-GCCcore-14.3.0-serial/include
        HDF5_LIBS ?= -L/p/software/default/stages/2026/software/HDF5/1.14.6-GCCcore-14.3.0-serial/lib -lhdf5
        CUDA_ARCH ?= sm_80
endif

ifeq ($(SYSTYPE),HorekaGH200)
# HorekaFTP (GH200): module load NVHPC/24.9-CUDA-12.6.0 HDF5/1.14.5-gompi-2024a
	CXX_RELEASE = nvc++
	HDF5_CFLAGS ?= -I/software/easybuild/software/HDF5/1.14.5-gompi-2024a/include
	HDF5_LIBS ?= -L/software/easybuild/software/HDF5/1.14.5-gompi-2024a/lib -lhdf5
	CUDA_ARCH ?= sm_90
endif

####################################################################################
# ADD YOUR SYSTEM TYPE, COMPILER AND HDF5 PATHS HERE IF NOT SUPPORTED
# ifeq ($(SYSTYPE),YourSystype)
# ...
# endif
####################################################################################

# HDF5
CXXFLAGS += $(HDF5_CFLAGS)
LDFLAGS += $(HDF5_LIBS)

# ============================================================
# capture git commit for banner (only Makefile + src/ count as dirty)
# ============================================================
GIT_DIRTY_PATHS := Makefile src

# short commit hash
GIT_HASH := $(shell git rev-parse --short HEAD 2>/dev/null)

# -dirty if tracked changes in Makefile/src or untracked files in src/
GIT_DIRTY := $(shell { ! git diff-index --quiet HEAD -- $(GIT_DIRTY_PATHS) 2>/dev/null || [ -n "$$(git ls-files --others --exclude-standard -- src 2>/dev/null)" ]; } && echo -dirty)

GIT_COMMIT := $(GIT_HASH)$(GIT_DIRTY)
ifeq ($(GIT_COMMIT),)
	GIT_COMMIT := unknown
endif
DEFINES += '-DGIT_COMMIT="$(GIT_COMMIT)"'

# +adds/-dels diffstat over the same paths (untracked src/ files counted as adds)
ifeq ($(GIT_DIRTY),-dirty)
	GIT_DIFFSTAT := $(shell { git diff HEAD --numstat -- $(GIT_DIRTY_PATHS) 2>/dev/null; git ls-files --others --exclude-standard -- src 2>/dev/null | while IFS= read -r f; do printf "%d\t0\n" "$$(wc -l < "$$f")"; done; } | awk '{a+=$$1; d+=$$2} END {printf "+%d -%d", a+0, d+0}')
	DEFINES += '-DGIT_DIFFSTAT="$(GIT_DIFFSTAT)"'
endif

# add config defines to compilation flags
CXXFLAGS += $(CONFIG_DEFINES) $(DEFINES)

# ============================================================
#  set up CUDA/MPI compilers
# ============================================================
ifeq ($(CUDA_ENABLED),CUDA)
        CXX = nvcc
        CXXFLAGS += -arch=$(CUDA_ARCH)
else
        CXX = ${CXX_RELEASE}
endif

ifndef CXX
	$(error SYSTYPE not recognized.)
endif

# optional: use MPI; route host compiler thorugh mpicxx
ifeq ($(MPI_ENABLED),USE_MPI)
        MPICXX ?= mpicxx
        export OMPI_CXX := $(CXX_RELEASE)
        export MPICH_CXX := $(CXX_RELEASE)

        ifeq ($(CUDA_ENABLED),CUDA)
                CXXFLAGS += -ccbin $(MPICXX)
                LDFLAGS  += -ccbin $(MPICXX)
        else
                CXX = $(MPICXX)
        endif
        MPI_MESSAGE = MPI enabled
else
        MPI_MESSAGE = MPI disabled
endif


# ============================================================
# Default target: build executable and print build summary
# ============================================================
all: $(TARGET)
	@echo "=========================================="
	@echo "Build complete! Executable: $(TARGET)"
	@echo "SYSTYPE: $(SYSTYPE)"
	@echo "Compiler: $(CXX)"
	@echo "Mode: $(BUILD_MODE_MESSAGE)"
	@echo "GPU: $(CUDA_MESSAGE)"
	@echo "OpenMP: $(OPENMP_MESSAGE)"
	@echo "MPI: $(MPI_MESSAGE)"
	@echo "=========================================="
ifeq ($(MPI_ENABLED),USE_MPI)
	@echo "Run with: mpirun -np [nranks] ./$(TARGET) [param.txt] [restart flag]"
else
	@echo "Run with: ./$(TARGET) [param.txt] [restart flag]"
endif

# ============================================================
# Linking
# ============================================================
ifeq ($(CUDA_ENABLED),CUDA)
# CUDA: two-step link (device link + host link)
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) -arch=$(CUDA_ARCH) -dlink $(OBJECTS) -o $(BUILD_DIR)/dlink.o $(LDFLAGS)
	$(CXX) -arch=$(CUDA_ARCH) $(OBJECTS) $(BUILD_DIR)/dlink.o -o $@ $(LDFLAGS) -lcudart
else
# CPU_DEBUG: single-step link with host compiler
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
endif

# ============================================================
# Compile sources
# ============================================================

# main entry point
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# global state and logging
$(BUILD_DIR)/globals.o: $(GLOBAL_DIR)/globals.cu $(GLOBAL_DIR)/globals.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/log.o: $(GLOBAL_DIR)/log.cu $(GLOBAL_DIR)/log.h $(MPI_DIR)/mpi_compat.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# I/O (HDF5 read/write)
$(BUILD_DIR)/input.o: $(IO_DIR)/input.cu $(IO_DIR)/input.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/output.o: $(IO_DIR)/output.cu $(IO_DIR)/output.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# k-nearest-neighbour search
$(BUILD_DIR)/knn.o: $(KNN_DIR)/knn.cu $(KNN_DIR)/knn.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# simulation initialization
$(BUILD_DIR)/begrun.o: $(BEGRUN_DIR)/begrun.cu $(BEGRUN_DIR)/begrun.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# voronoi mesh construction
$(BUILD_DIR)/voronoi.o: $(VORONOI_DIR)/voronoi.cu $(VORONOI_DIR)/voronoi.h $(VORONOI_DIR)/internal.h $(VORONOI_DIR)/alloc.cu $(VORONOI_DIR)/build.cu $(VORONOI_DIR)/cell.cu $(VORONOI_DIR)/cell.h $(VORONOI_DIR)/fallback.cu $(VORONOI_DIR)/geometry.cu $(VORONOI_DIR)/geometry.h $(VORONOI_DIR)/ghosts.cu | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/moving.o: $(VORONOI_DIR)/moving.cu $(VORONOI_DIR)/voronoi.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# finite-volume hydro solver (incl. Riemann solver)
$(BUILD_DIR)/finite_volume_solver.o: $(HYDRO_DIR)/finite_volume_solver.cu $(HYDRO_DIR)/finite_volume_solver.h $(HYDRO_DIR)/riemann.cu $(HYDRO_DIR)/riemann.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# gradient estimation
$(BUILD_DIR)/gradients.o: $(GRADIENTS_DIR)/gradients.cu $(GRADIENTS_DIR)/gradients.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# profiler
$(BUILD_DIR)/profiler.o: $(PROFILER_DIR)/profiler.cu $(PROFILER_DIR)/profiler.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# MPI: init, domain decomposition, halo exchange, particle migration
$(BUILD_DIR)/mpi_compat.o: $(MPI_DIR)/mpi_compat.cu $(MPI_DIR)/mpi_compat.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/decomp.o: $(MPI_DIR)/decomp.cu $(MPI_DIR)/decomp.h $(MPI_DIR)/mpi_compat.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/halo.o: $(MPI_DIR)/halo.cu $(MPI_DIR)/halo.h \
    $(MPI_DIR)/halo_internal.cu $(MPI_DIR)/halo_init.cu \
    $(MPI_DIR)/halo_build.cu $(MPI_DIR)/halo_exchange.cu \
    $(MPI_DIR)/decomp.h $(MPI_DIR)/mpi_compat.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/migrate.o: $(MPI_DIR)/migrate.cu $(MPI_DIR)/migrate.h $(MPI_DIR)/decomp.h $(MPI_DIR)/halo.h $(MPI_DIR)/mpi_compat.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# ============================================================
# Utility targets
# ============================================================

# create build directory if missing
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# optional: also run the programm
run: $(TARGET)
	@echo "Running application..."
	@./$(TARGET)

# pull in auto-generated header dependencies (.d files from -MMD -MP)
-include $(OBJECTS:.o=.d)

# clean build artifacts
clean:
	@echo "Cleaning build files..."
	@rm -f $(OBJECTS) $(BUILD_DIR)/dlink.o $(TARGET)
	@rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
