# parallel build is default
MAKEFLAGS += -j

# compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++11 -O2
INCLUDES = -Isrc -Isrc/global

# directories
SRC_DIR = src
BUILD_DIR = build
OUTPUT_DIR = output
GLOBAL_DIR = $(SRC_DIR)/global
IO_DIR = $(SRC_DIR)/io
KNN_DIR = $(SRC_DIR)/knn
BEGRUN_DIR = $(SRC_DIR)/begrun
HDF5_LIB_DIR = libs/hdf5/lib

# source files
MAIN_SRC = $(SRC_DIR)/main.cpp
GLOBAL_SRC = $(GLOBAL_DIR)/allvars.cpp
IO_SRC = $(IO_DIR)/input.cpp $(IO_DIR)/output.cpp
KNN_SRC = $(KNN_DIR)/knn.cpp
BEGRUN_SRC = $(BEGRUN_DIR)/begrun.cpp
SOURCES = $(MAIN_SRC) $(GLOBAL_SRC) $(IO_SRC) $(KNN_SRC) $(BEGRUN_SRC)

# object files
MAIN_OBJ = $(BUILD_DIR)/main.o
GLOBAL_OBJ = $(BUILD_DIR)/allvars.o
IO_OBJ = $(BUILD_DIR)/input.o $(BUILD_DIR)/output.o
KNN_OBJ = $(BUILD_DIR)/knn.o
BEGRUN_OBJ = $(BUILD_DIR)/begrun.o
OBJECTS = $(MAIN_OBJ) $(GLOBAL_OBJ) $(IO_OBJ) $(KNN_OBJ) $(BEGRUN_OBJ)

# name of executable (see: https://en.wikipedia.org/wiki/Proteus :D)
TARGET = ProteusGPU

# load Config.sh options and convert them to -D flags
SHELL := /bin/bash
CONFIG_DEFINES := $(shell grep -v "^\#" Config.sh | grep -v "^$$" | grep -v "^!" | awk 'NF {print "-D" $$1}')

# system-specific includes
SYSTYPE ?= $(shell uname -s)
-include Makefile.systype

ifeq ($(SYSTYPE),Ubuntu)
	HDF5_CFLAGS ?= -I/usr/include/hdf5/serial
	HDF5_LIBS ?= -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
endif

ifeq ($(SYSTYPE),macOS)
	HDF5_CFLAGS ?= -I/opt/homebrew/opt/hdf5/include
	HDF5_LIBS ?= -L/opt/homebrew/opt/hdf5/lib -lhdf5
endif

ifeq ($(SYSTYPE),MPCDF)
        HDF5_CFLAGS ?= -I${HDF5_HOME}/include
        HDF5_LIBS ?= -L${HDF5_HOME}/lib -lhdf5
endif

# ADD YOUR SYSTEM TYPE AND HDF5 PATHS HERE IF NOT SUPPORTED
# ifeq ($(SYSTYPE),YourSystype)
# ...
# endif

# check if HDF5 is enabled in Config.sh
ifneq (,$(findstring USE_HDF5,$(CONFIG_DEFINES)))
	HAS_HDF5 = 1
	CXXFLAGS += $(HDF5_CFLAGS)
	LDFLAGS += $(HDF5_LIBS)
endif

# add config defines to compilation flags
CXXFLAGS += $(CONFIG_DEFINES)

# default target
all: $(TARGET)
	@echo "Build complete! Executable: $(TARGET)"
	@echo "Run with: ./$(TARGET)"

$(TARGET): $(OBJECTS) | $(BUILD_DIR)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

# compile sources
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/allvars.o: $(GLOBAL_DIR)/allvars.cpp $(GLOBAL_DIR)/allvars.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/input.o: $(IO_DIR)/input.cpp $(IO_DIR)/input.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/output.o: $(IO_DIR)/output.cpp $(IO_DIR)/output.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/knn.o: $(KNN_DIR)/knn.cpp $(KNN_DIR)/knn.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/begrun.o: $(BEGRUN_DIR)/begrun.cpp $(BEGRUN_DIR)/begrun.h | $(BUILD_DIR)
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

# clean build files
clean:
	@echo "Cleaning build files..."
	@rm -f $(OBJECTS) $(TARGET)
	@rm -rf $(BUILD_DIR)/*.o
