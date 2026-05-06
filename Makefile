# parallel build is default
MAKEFLAGS += -j

# user-overridable paths (override on command line, e.g. make CONFIG=foo/Config.sh BUILD_DIR=foo/build EXEC=foo/ProteusGPU)
CONFIG ?= Config.sh
BUILD_DIR ?= build
EXEC ?= ProteusGPU

# load Config.sh options and convert them to -D flags FIRST (before compiler setup)
SHELL := /bin/bash
CONFIG_DEFINES := $(shell grep -v "^\#" $(CONFIG) | grep -v "^$$" | grep -v "^!" | awk 'NF {print "-D" $$1}')
DEFINES :=

# system-specific includes
ifdef SYSTYPE
        SYSTYPE := $(SYSTYPE)
else
        SYSTYPE ?= $(shell uname -s)
        -include Makefile.systype
endif

# Check CUDA / PROFILING
CUDA_ENABLED := $(findstring CUDA,$(CONFIG_DEFINES))
PROFILING_ENABLED := $(findstring ENABLE_PROFILING,$(CONFIG_DEFINES))

# ============================================================
# CUDA mode: nvcc compiler
# ============================================================
ifeq ($(CUDA_ENABLED),CUDA)

CXXFLAGS = --compiler-options -Wall,-Wextra,-Wno-unknown-pragmas -std=c++14
CXXFLAGS += --expt-relaxed-constexpr
CXXFLAGS += -dc -O3 --prec-div=false --prec-sqrt=false --ftz=true --fmad=true
LDFLAGS =
BUILD_MODE_MESSAGE = CUDA RELEASE

ifneq (,$(findstring USE_OPENMP,$(CONFIG_DEFINES)))
	CXXFLAGS += --compiler-options -fopenmp
	LDFLAGS += --compiler-options -fopenmp
	OPENMP_MESSAGE = OpenMP enabled
else
	OPENMP_MESSAGE = OpenMP disabled
endif

CUDA_MESSAGE = CUDA enabled

# Link NVTX for profiling annotations (Nsight Systems)
ifeq ($(PROFILING_ENABLED),ENABLE_PROFILING)
	LDFLAGS += -lnvToolsExt
endif

# ============================================================
# CPU_DEBUG mode: host compiler (g++/clang++)
# ============================================================
else

CXXFLAGS = -Wall -Wextra -std=c++14 -O3
LDFLAGS =
BUILD_MODE_MESSAGE = RELEASE

# g++ does not recognize .cu extension — treat as C++
CXXFLAGS += -x c++

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

# directories
SRC_DIR = src
GLOBAL_DIR = $(SRC_DIR)/global
IO_DIR = $(SRC_DIR)/io
KNN_DIR = $(SRC_DIR)/knn
BEGRUN_DIR = $(SRC_DIR)/begrun
VORONOI_DIR = $(SRC_DIR)/voronoi
HYDRO_DIR = $(SRC_DIR)/hydro
GRADIENTS_DIR = $(SRC_DIR)/gradients
PROFILER_DIR = $(SRC_DIR)/profiler

# object files
MAIN_OBJ = $(BUILD_DIR)/main.o
GLOBAL_OBJ = $(BUILD_DIR)/globals.o
IO_OBJ = $(BUILD_DIR)/input.o $(BUILD_DIR)/output.o
KNN_OBJ = $(BUILD_DIR)/knn.o
BEGRUN_OBJ = $(BUILD_DIR)/begrun.o
VORONOI_OBJ = $(BUILD_DIR)/voronoi.o $(BUILD_DIR)/periodic_mesh.o
HYDRO_OBJ = $(BUILD_DIR)/finite_volume_solver.o
GRADIENTS_OBJ = $(BUILD_DIR)/gradients.o
PROFILER_OBJ = $(BUILD_DIR)/profiler.o
OBJECTS = $(MAIN_OBJ) $(GLOBAL_OBJ) $(IO_OBJ) $(KNN_OBJ) $(BEGRUN_OBJ) $(VORONOI_OBJ) $(HYDRO_OBJ) $(GRADIENTS_OBJ) $(PROFILER_OBJ)

# name of executable
TARGET = $(EXEC)

# System Types
ifeq ($(SYSTYPE),Ubuntu)
	CXX_RELEASE = g++
	HDF5_CFLAGS ?= -I/usr/include/hdf5/serial
	HDF5_LIBS ?= -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
	CUDA_ARCH ?= sm_89
endif

ifeq ($(SYSTYPE),macOS)
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

ifeq ($(SYSTYPE),HorekaGH200)
# HorekaFTP (GH200): module load NVHPC/24.9-CUDA-12.6.0 HDF5/1.14.5-gompi-2024a
	CXX_RELEASE = nvc++
	HDF5_CFLAGS ?= -I/software/easybuild/software/HDF5/1.14.5-gompi-2024a/include
	HDF5_LIBS ?= -L/software/easybuild/software/HDF5/1.14.5-gompi-2024a/lib -lhdf5
	CUDA_ARCH ?= sm_90
endif

# ADD YOUR SYSTEM TYPE AND HDF5 PATHS HERE IF NOT SUPPORTED
# ifeq ($(SYSTYPE),YourSystype)
# ...
# endif

# Determine compiler based on CUDA/CPU_DEBUG mode and platform
ifeq ($(CUDA_ENABLED),CUDA)
        CXX = nvcc
        CXXFLAGS += -arch=$(CUDA_ARCH)
else
        CXX = ${CXX_RELEASE}
endif

ifndef CXX
	$(error SYSTYPE not recognized.)
endif

# HDF5
CXXFLAGS += $(HDF5_CFLAGS)
LDFLAGS += $(HDF5_LIBS)

# add config defines to compilation flags
CXXFLAGS += $(CONFIG_DEFINES) $(DEFINES)

# default target
all: $(TARGET)
	@echo "=========================================="
	@echo "Build complete! Executable: $(TARGET)"
	@echo "SYSTYPE: $(SYSTYPE)"
	@echo "Compiler: $(CXX)"
	@echo "Mode: $(BUILD_MODE_MESSAGE)"
	@echo "GPU: $(CUDA_MESSAGE)"
	@echo "OpenMP: $(OPENMP_MESSAGE)"
	@echo "=========================================="
	@echo "Run with: ./$(TARGET) [param.txt] [restart flag]"

# ---- Linking ----
ifeq ($(CUDA_ENABLED),CUDA)
# CUDA: two-step link (device link + host link)
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) -arch=$(CUDA_ARCH) -dlink $(OBJECTS) -o $(BUILD_DIR)/dlink.o $(LDFLAGS)
	$(CXX) -arch=$(CUDA_ARCH) $(OBJECTS) $(BUILD_DIR)/dlink.o -o $@ $(LDFLAGS) -lcudart
else
$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)
endif

# compile sources
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cu | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/globals.o: $(GLOBAL_DIR)/globals.cu $(GLOBAL_DIR)/globals.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/input.o: $(IO_DIR)/input.cu $(IO_DIR)/input.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/output.o: $(IO_DIR)/output.cu $(IO_DIR)/output.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/knn.o: $(KNN_DIR)/knn.cu $(KNN_DIR)/knn.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/begrun.o: $(BEGRUN_DIR)/begrun.cu $(BEGRUN_DIR)/begrun.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/voronoi.o: $(VORONOI_DIR)/voronoi.cu $(VORONOI_DIR)/voronoi.h $(VORONOI_DIR)/cell.cu $(VORONOI_DIR)/cell.h $(VORONOI_DIR)/geometry.cu $(VORONOI_DIR)/geometry.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/periodic_mesh.o: $(VORONOI_DIR)/periodic_mesh.cu $(VORONOI_DIR)/periodic_mesh.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/finite_volume_solver.o: $(HYDRO_DIR)/finite_volume_solver.cu $(HYDRO_DIR)/finite_volume_solver.h $(HYDRO_DIR)/riemann.cu $(HYDRO_DIR)/riemann.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/gradients.o: $(GRADIENTS_DIR)/gradients.cu $(GRADIENTS_DIR)/gradients.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/profiler.o: $(PROFILER_DIR)/profiler.cu $(PROFILER_DIR)/profiler.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# create directories if missing
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# optional also run the programm
run: $(TARGET)
	@echo "Running application..."
	@./$(TARGET)

# automatic header dependencies
-include $(OBJECTS:.o=.d)

# clean build files
clean:
	@echo "Cleaning build files..."
	@rm -f $(OBJECTS) $(BUILD_DIR)/dlink.o $(TARGET)
	@rm -rf $(BUILD_DIR)/*.o $(BUILD_DIR)/*.d
