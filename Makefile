# parallel build is default
MAKEFLAGS += -j

# load Config.sh options and convert them to -D flags FIRST (before compiler setup)
SHELL := /bin/bash
CONFIG_DEFINES := $(shell grep -v "^\#" Config.sh | grep -v "^$$" | grep -v "^!" | awk 'NF {print "-D" $$1}')
DEFINES := 

# system-specific includes
ifdef SYSTYPE
        SYSTYPE := $(SYSTYPE)
else
        SYSTYPE ?= $(shell uname -s)
        -include Makefile.systype
endif

# Check for DEBUG_MODE FIRST (before compiler setup)
DEBUG_MODE_ENABLED := $(findstring DEBUG_MODE,$(CONFIG_DEFINES))

# Base compiler flags
CXXFLAGS = -Wall -Wextra -std=c++14

# Check for DEBUG_MODE and set optimization flags
ifeq ($(DEBUG_MODE_ENABLED),DEBUG_MODE)
	# Debug mode: enable AddressSanitizer
	CXXFLAGS += -O0 -g -fsanitize=address
	LDFLAGS = -fsanitize=address
	BUILD_MODE_MESSAGE = DEBUG (AddressSanitizer enabled)
else
	# Release mode: optimize for performance
	CXXFLAGS += -O3
	LDFLAGS = 
	BUILD_MODE_MESSAGE = RELEASE (optimized)
endif

# Check if OpenMP is enabled
ifneq (,$(findstring USE_OPENMP,$(CONFIG_DEFINES)))
	CXXFLAGS += -fopenmp
	LDFLAGS += -fopenmp
	OPENMP_MESSAGE = OpenMP enabled
else
	OPENMP_MESSAGE = OpenMP disabled
endif

CXXFLAGS += -MMD -MP
INCLUDES = -Isrc -Isrc/global

# directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = output
GLOBAL_DIR = $(SRC_DIR)/global
IO_DIR = $(SRC_DIR)/io
KNN_DIR = $(SRC_DIR)/knn
BEGRUN_DIR = $(SRC_DIR)/begrun
VORONOI_DIR = $(SRC_DIR)/voronoi
HYDRO_DIR = $(SRC_DIR)/hydro
GRADIENTS_DIR = $(SRC_DIR)/gradients
PROFILER_DIR = $(SRC_DIR)/profiler
HDF5_LIB_DIR = libs/hdf5/lib

# source files
MAIN_SRC = $(SRC_DIR)/main.cpp
GLOBAL_SRC = $(GLOBAL_DIR)/globals.cpp
IO_SRC = $(IO_DIR)/input.cpp $(IO_DIR)/output.cpp
KNN_SRC = $(KNN_DIR)/knn.cpp
BEGRUN_SRC = $(BEGRUN_DIR)/begrun.cpp
VORONOI_SRC = $(VORONOI_DIR)/voronoi.cpp $(VORONOI_DIR)/cell.cpp $(VORONOI_DIR)/periodic_mesh.cpp $(VORONOI_DIR)/geometry.cpp
HYDRO_SRC = $(HYDRO_DIR)/finite_volume_solver.cpp $(HYDRO_DIR)/riemann.cpp
GRADIENTS_SRC = $(GRADIENTS_DIR)/gradients.cpp
PROFILER_SRC = $(PROFILER_DIR)/profiler.cpp
SOURCES = $(MAIN_SRC) $(GLOBAL_SRC) $(IO_SRC) $(KNN_SRC) $(BEGRUN_SRC) $(VORONOI_SRC) $(HYDRO_SRC) $(GRADIENTS_SRC) $(PROFILER_SRC)

# object files
MAIN_OBJ = $(BUILD_DIR)/main.o
GLOBAL_OBJ = $(BUILD_DIR)/globals.o
IO_OBJ = $(BUILD_DIR)/input.o $(BUILD_DIR)/output.o
KNN_OBJ = $(BUILD_DIR)/knn.o
BEGRUN_OBJ = $(BUILD_DIR)/begrun.o
VORONOI_OBJ = $(BUILD_DIR)/voronoi.o $(BUILD_DIR)/cell.o $(BUILD_DIR)/periodic_mesh.o $(BUILD_DIR)/geometry.o
HYDRO_OBJ = $(BUILD_DIR)/finite_volume_solver.o $(BUILD_DIR)/riemann.o
GRADIENTS_OBJ = $(BUILD_DIR)/gradients.o
PROFILER_OBJ = $(BUILD_DIR)/profiler.o
OBJECTS = $(MAIN_OBJ) $(GLOBAL_OBJ) $(IO_OBJ) $(KNN_OBJ) $(BEGRUN_OBJ) $(VORONOI_OBJ) $(HYDRO_OBJ) $(GRADIENTS_OBJ) $(PROFILER_OBJ)

# name of executable (see: https://en.wikipedia.org/wiki/Proteus :D)
TARGET = ProteusGPU

ifeq ($(SYSTYPE),Ubuntu)
	CXX_DEBUG = g++
	CXX_RELEASE = g++
	HDF5_CFLAGS ?= -I/usr/include/hdf5/serial
	HDF5_LIBS ?= -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
endif

ifeq ($(SYSTYPE),macOS)
	CXX_DEBUG = clang++ # better address sanitizer support
	CXX_RELEASE = g++-15
	HDF5_CFLAGS ?= -I/opt/homebrew/opt/hdf5/include
	HDF5_LIBS ?= -L/opt/homebrew/opt/hdf5/lib -lhdf5
endif

ifeq ($(SYSTYPE),MPCDF)
	CXX_DEBUG = g++
	CXX_RELEASE = g++
        HDF5_CFLAGS ?= -I${HDF5_HOME}/include
        HDF5_LIBS ?= -L${HDF5_HOME}/lib -lhdf5
endif

ifeq ($(SYSTYPE),HorekaGH200)
	# module load NVHPC/24.9-CUDA-12.6.0 HDF5/1.14.5-gompi-2024a
	CXX_DEBUG = nvc++
	CXX_RELEASE = nvc++
	HDF5_CFLAGS ?= -I/software/easybuild/software/HDF5/1.14.5-gompi-2024a/include
	HDF5_LIBS ?= -L/software/easybuild/software/HDF5/1.14.5-gompi-2024a/lib -lhdf5
endif

# ADD YOUR SYSTEM TYPE AND HDF5 PATHS HERE IF NOT SUPPORTED
# ifeq ($(SYSTYPE),YourSystype)
# ...
# endif

# Determine compiler based on DEBUG_MODE and platform
ifeq ($(DEBUG_MODE_ENABLED),DEBUG_MODE)
        CXX = ${CXX_DEBUG}
else
        CXX = ${CXX_RELEASE}
endif

ifndef CXX
	$(error SYSTYPE not recognized.)
endif

# check if HDF5 is enabled in Config.sh
ifneq (,$(findstring USE_HDF5,$(CONFIG_DEFINES)))
	HAS_HDF5 = 1
	CXXFLAGS += $(HDF5_CFLAGS)
	LDFLAGS += $(HDF5_LIBS)
endif

# add config defines to compilation flags
CXXFLAGS += $(CONFIG_DEFINES) $(DEFINES)

# default target
all: $(TARGET)
	@echo "=========================================="
	@echo "Build complete! Executable: $(TARGET)"
	@echo "SYSTYPE: $(SYSTYPE)"
	@echo "Compiler: $(CXX)"
	@echo "Mode: $(BUILD_MODE_MESSAGE)"
	@echo "OpenMP: $(OPENMP_MESSAGE)"
	@echo "=========================================="
	@echo "Run with: ./$(TARGET) [param.txt] [restart flag]"

$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

# compile sources
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/globals.o: $(GLOBAL_DIR)/globals.cpp $(GLOBAL_DIR)/globals.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/input.o: $(IO_DIR)/input.cpp $(IO_DIR)/input.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/output.o: $(IO_DIR)/output.cpp $(IO_DIR)/output.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/knn.o: $(KNN_DIR)/knn.cpp $(KNN_DIR)/knn.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/begrun.o: $(BEGRUN_DIR)/begrun.cpp $(BEGRUN_DIR)/begrun.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/voronoi.o: $(VORONOI_DIR)/voronoi.cpp $(VORONOI_DIR)/voronoi.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/cell.o: $(VORONOI_DIR)/cell.cpp $(VORONOI_DIR)/cell.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/periodic_mesh.o: $(VORONOI_DIR)/periodic_mesh.cpp $(VORONOI_DIR)/periodic_mesh.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/geometry.o: $(VORONOI_DIR)/geometry.cpp $(VORONOI_DIR)/geometry.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/finite_volume_solver.o: $(HYDRO_DIR)/finite_volume_solver.cpp $(HYDRO_DIR)/finite_volume_solver.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/riemann.o: $(HYDRO_DIR)/riemann.cpp $(HYDRO_DIR)/riemann.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/gradients.o: $(GRADIENTS_DIR)/gradients.cpp $(GRADIENTS_DIR)/gradients.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/profiler.o: $(PROFILER_DIR)/profiler.cpp $(PROFILER_DIR)/profiler.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# create directories if missing
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

# optional also run the programm
run: $(TARGET)
	@echo "Running application..."
	@./$(TARGET)

# automatic header dependencies
-include $(OBJECTS:.o=.d)

# clean build files
clean:
	@echo "Cleaning build files..."
	@rm -f $(OBJECTS) $(TARGET)
	@rm -rf $(BUILD_DIR)/*.o
