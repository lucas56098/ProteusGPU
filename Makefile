# Builds Proteus given a Config.sh and the SYSTYPE in Makefile.systype.
#
#   make           the executable
#   make test      builds and runs the test program of every module
#   make check     checks the includes against the module table, and the format of every file
#   make clean     removes the build folder and the executable
#
# CONFIG, BUILD_DIR and EXEC can be set on the command line, e.g. make CONFIG=other_config.sh

CONFIG    ?= Config.sh
BUILD_DIR ?= build
EXEC      ?= Proteus

MAKEFLAGS += -j

# fist word of every line in CONFIG that doesnt start with #
SWITCHES := $(shell awk '!/^[[:space:]]*#/ && NF {print $$1}' $(CONFIG))
is_on     = $(filter $(1),$(SWITCHES))

# ==========================================================
# machine setup
# ==========================================================

ifndef SYSTYPE
-include Makefile.systype
endif

# Ubuntu Machine
ifeq ($(SYSTYPE),Ubuntu)
    HOST_CXX              := g++
    MPI_CXX               := mpicxx
    GPU_ARCH              := sm_89
    HDF5_SERIAL_INCLUDE   := -I/usr/include/hdf5/serial
    HDF5_SERIAL_LIBRARY   := -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5
    HDF5_PARALLEL_INCLUDE := -I/usr/include/hdf5/openmpi
    HDF5_PARALLEL_LIBRARY := -L/usr/lib/x86_64-linux-gnu/hdf5/openmpi -lhdf5
endif

# Jupiter BOOSTER (JSC): 4xGH200 chips per node (sm_90)
#       module load GCC ParaStationMPI CUDA HDF5 (has GPU aware MPI + parallel HDF5)
ifeq ($(SYSTYPE),JUPITER)
    HOST_CXX              := g++
    MPI_CXX               := mpicxx
    GPU_ARCH              := sm_90
    HDF5_SERIAL_INCLUDE   := -I$(EBROOTHDF5)/include
    HDF5_SERIAL_LIBRARY   := -L$(EBROOTHDF5)/lib -lhdf5
    HDF5_PARALLEL_INCLUDE := -I$(EBROOTHDF5)/include
    HDF5_PARALLEL_LIBRARY := -L$(EBROOTHDF5)/lib -lhdf5
endif

# macOS: CPU only, with g++-15 as apple clang has no OpenMP
#       brew install hdf5 (and hdf5-mpi for USE_MPI, its serial keg stays in the Cellar when unlinked)
ifeq ($(SYSTYPE),macOS)
    HOST_CXX              := g++-15
    MPI_CXX               := mpicxx
    GPU_ARCH              :=
    HDF5_SERIAL_PREFIX    := $(shell test -d /opt/homebrew/opt/hdf5 && echo /opt/homebrew/opt/hdf5 || ls -d /opt/homebrew/Cellar/hdf5/*/ 2>/dev/null | head -n1)
    HDF5_SERIAL_INCLUDE   := -I$(HDF5_SERIAL_PREFIX)/include
    HDF5_SERIAL_LIBRARY   := -L$(HDF5_SERIAL_PREFIX)/lib -lhdf5
    HDF5_PARALLEL_INCLUDE := -I/opt/homebrew/opt/hdf5-mpi/include
    HDF5_PARALLEL_LIBRARY := -L/opt/homebrew/opt/hdf5-mpi/lib -lhdf5
endif

# MPCDF and BinAC2 (A100); USE_MPI needs a parallel HDF5 module, e.g. VERA: hdf5-mpi/1.12.2
#       VERA: module load gcc/15 hdf5-serial/1.12.2 cuda/13.0
#       BinAC2: module load compiler/gnu/14.2 lib/hdf5/1.12-gnu-14.2 devel/cuda/13.0
ifeq ($(SYSTYPE),MPCDF)
    HOST_CXX              := g++
    MPI_CXX               := mpicxx
    GPU_ARCH              := sm_80
    HDF5_SERIAL_INCLUDE   := -I$(HDF5_HOME)/include
    HDF5_SERIAL_LIBRARY   := -L$(HDF5_HOME)/lib -lhdf5
    HDF5_PARALLEL_INCLUDE := -I$(HDF5_HOME)/include
    HDF5_PARALLEL_LIBRARY := -L$(HDF5_HOME)/lib -lhdf5
endif

# JUWELS Booster (A100)
#       module load GCC/14.3.0 ParaStationMPI/5.13.0-1 CUDA/13 HDF5/1.14.6 (has GPU aware MPI)
#       without GPU aware MPI: module swap MPI-settings/CUDA MPI-settings/UCX
ifeq ($(SYSTYPE),JUWELS)
    HOST_CXX              := g++
    MPI_CXX               := mpicxx
    GPU_ARCH              := sm_80
    HDF5_SERIAL_INCLUDE   := -I$(EBROOTHDF5)/include
    HDF5_SERIAL_LIBRARY   := -L$(EBROOTHDF5)/lib -lhdf5
    HDF5_PARALLEL_INCLUDE := -I$(EBROOTHDF5)/include
    HDF5_PARALLEL_LIBRARY := -L$(EBROOTHDF5)/lib -lhdf5
endif

# HoreKa FTP (GH200)
#       module load HDF5/1.14.5-gompi-2024a OpenMPI/5.0.3-NVHPC-24.9-CUDA-12.6.0 (has GPU aware MPI)
ifeq ($(SYSTYPE),HorekaGH200)
    HOST_CXX              := g++
    MPI_CXX               := mpicxx
    GPU_ARCH              := sm_90
    HDF5_SERIAL_INCLUDE   := -I$(EBROOTHDF5)/include
    HDF5_SERIAL_LIBRARY   := -L$(EBROOTHDF5)/lib -lhdf5
    HDF5_PARALLEL_INCLUDE := -I$(EBROOTHDF5)/include
    HDF5_PARALLEL_LIBRARY := -L$(EBROOTHDF5)/lib -lhdf5
endif

# HoreKa2 Jade (GB200)
#       module load nvidia-compilers/25.9-CUDA-12.9.1 HDF5/1.14.6-NVHPC-25.9-CUDA-12.9.1 gompi/2025b
ifeq ($(SYSTYPE),Horeka2)
    HOST_CXX              := g++
    MPI_CXX               := mpicxx
    GPU_ARCH              := sm_100f
    HDF5_SERIAL_INCLUDE   := -I$(EBROOTHDF5)/include
    HDF5_SERIAL_LIBRARY   := -L$(EBROOTHDF5)/lib -lhdf5
    HDF5_PARALLEL_INCLUDE := -I$(EBROOTHDF5)/include
    HDF5_PARALLEL_LIBRARY := -L$(EBROOTHDF5)/lib -lhdf5
endif

#########################################
# INCLUDE YOUR SYSTYPE HERE
#
#   ifeq ($(SYSTYPE),YourSystype)
#       HOST_CXX              :=
#       MPI_CXX               :=
#       GPU_ARCH              :=
#       HDF5_SERIAL_INCLUDE   :=
#       HDF5_SERIAL_LIBRARY   :=
#       HDF5_PARALLEL_INCLUDE :=
#       HDF5_PARALLEL_LIBRARY :=
#   endif
#
#########################################

ifeq ($(HOST_CXX),)
    $(error Unknown SYSTYPE "$(SYSTYPE)". Add a block for it to the Makefile)
endif


# ==========================================================
# compiler and flags
# ==========================================================

# with MPI the host compiler is called through the MPI wrapper, and HDF5 is the parallel one
ifneq ($(call is_on,USE_MPI),)
    HOST := $(MPI_CXX)
    export OMPI_CXX  := $(HOST_CXX)
    export MPICH_CXX := $(HOST_CXX)
    HDF5_INCLUDE := $(HDF5_PARALLEL_INCLUDE)
    HDF5_LIBRARY := $(HDF5_PARALLEL_LIBRARY)
else
    HOST := $(HOST_CXX)
    HDF5_INCLUDE := $(HDF5_SERIAL_INCLUDE)
    HDF5_LIBRARY := $(HDF5_SERIAL_LIBRARY)
endif

# the same arithmetic on CPU and GPU; on the host: no contraction into fused multiply-add
HOST_FLAGS := -O3 -Wall -Wextra -Werror -ffp-contract=off
ifneq ($(call is_on,USE_OPENMP),)
    HOST_FLAGS += -fopenmp
endif

# on the GPU: no fused multiply-add, exact division and square root, no flushing of tiny numbers to zero; with
# CUDA_FAST_MATH all of these for speed. The host compiler reads .cu files as C++ only when told so with -x c++.
ifneq ($(call is_on,CUDA_FAST_MATH),)
    ifeq ($(call is_on,CUDA),)
        $(error Config.sh: CUDA_FAST_MATH needs CUDA)
    endif
    GPU_MATH := --fmad=true --prec-div=false --prec-sqrt=false --ftz=true
else
    GPU_MATH := --fmad=false --prec-div=true --prec-sqrt=true --ftz=false
endif
ifneq ($(call is_on,CUDA),)
    CXX         := nvcc
    GPU_FLAGS   := -arch=$(GPU_ARCH) -ccbin $(HOST) $(GPU_MATH)
    CXXFLAGS    := -std=c++17 $(GPU_FLAGS) --extended-lambda -Werror all-warnings $(addprefix -Xcompiler ,$(HOST_FLAGS))
    LDFLAGS     := $(GPU_FLAGS) $(addprefix -Xcompiler ,$(HOST_FLAGS))
    SOURCE_LANG :=
else
    CXX         := $(HOST)
    CXXFLAGS    := -std=c++17 $(HOST_FLAGS)
    LDFLAGS     := $(HOST_FLAGS)
    SOURCE_LANG := -x c++
endif

DEFINES := $(addprefix -D,$(SWITCHES))

# compiles one .cu file into an object, and lists the headers it includes in a .d file next to it
COMPILE = $(CXX) $(CXXFLAGS) $(DEFINES) -Isrc $(HDF5_INCLUDE) -MMD -MP -MF $(@:.o=.d)

# sources: every .cu file in src/, one object each
SOURCES := $(shell find src -name '*.cu')
OBJECTS := $(patsubst src/%.cu,$(BUILD_DIR)/%.o,$(SOURCES))

# tests: one program test_<module> per folder tests/<module>/
TEST_MODULES  := $(sort $(notdir $(patsubst %/,%,$(dir $(wildcard tests/*/*.cu)))))
TEST_PROGRAMS := $(addprefix $(BUILD_DIR)/tests/test_,$(TEST_MODULES))
TEST_OBJECTS  := $(patsubst tests/%.cu,$(BUILD_DIR)/tests/%.o,$(wildcard tests/*.cu tests/*/*.cu))
TEST_SHARED   := $(BUILD_DIR)/tests/main.o $(filter-out $(BUILD_DIR)/main.o,$(OBJECTS))

# to check includes against module table, and clang-format
CLANG_FORMAT ?= clang-format
FORMATTED    := $(filter-out tests/doctest.h,$(shell find src tests -name '*.cu' -o -name '*.h'))

# ==========================================================
# targets
# ==========================================================

.PHONY: all test check clean FORCE

all: $(EXEC)

$(EXEC): $(OBJECTS)
	$(CXX) $(LDFLAGS) $^ $(HDF5_LIBRARY) -o $@
	@echo "Built $(EXEC) on $(SYSTYPE) with: $(SWITCHES)"

test: $(TEST_PROGRAMS)
	@for program in $^; do echo "== $$program"; $$program || exit 1; done

# the link rule of one test program; written out below for every module
define test_program
$(BUILD_DIR)/tests/test_$(1): $(patsubst tests/%.cu,$(BUILD_DIR)/tests/%.o,$(wildcard tests/$(1)/*.cu)) $(TEST_SHARED)
	$$(CXX) $$(LDFLAGS) $$^ $$(HDF5_LIBRARY) -o $$@
endef
$(foreach module,$(TEST_MODULES),$(eval $(call test_program,$(module))))

# the check of src/ must pass
check:
	python3 tests/check_dependencies.py src
	python3 tests/check_dependencies.py --self-test tests/dependencies_bad/src
	$(CLANG_FORMAT) --dry-run --Werror $(FORMATTED)
	@echo "format check: every file is formatted"

# the commit in the hello message; only Makefile and src count as changes. begrun.o is built again when it changes
GIT_PATHS  := Makefile src
GIT_HASH   := $(shell git rev-parse --short HEAD 2>/dev/null)
GIT_DIRTY  := $(shell { ! git diff-index --quiet HEAD -- $(GIT_PATHS) 2>/dev/null || [ -n "$$(git ls-files --others --exclude-standard -- src 2>/dev/null)" ]; } && echo -dirty)
GIT_COMMIT := $(if $(GIT_HASH),$(GIT_HASH)$(GIT_DIRTY),unknown)
$(BUILD_DIR)/begrun/begrun.o: DEFINES += '-DGIT_COMMIT="$(GIT_COMMIT)"'
ifneq ($(GIT_DIRTY),)
    GIT_DIFFSTAT := $(shell { git diff HEAD --numstat -- $(GIT_PATHS) 2>/dev/null; git ls-files --others --exclude-standard -- src 2>/dev/null | while IFS= read -r f; do printf "%d\t0\n" "$$(wc -l < "$$f")"; done; } | awk '{a+=$$1; d+=$$2} END {printf "+%d -%d", a+0, d+0}')
    $(BUILD_DIR)/begrun/begrun.o: DEFINES += '-DGIT_DIFFSTAT="$(GIT_DIFFSTAT)"'
endif

$(BUILD_DIR)/begrun/begrun.o: $(BUILD_DIR)/commit_of_build
$(BUILD_DIR)/commit_of_build: FORCE
	@mkdir -p $(dir $@)
	@echo '$(GIT_COMMIT) $(GIT_DIFFSTAT)' | cmp -s - $@ || echo '$(GIT_COMMIT) $(GIT_DIFFSTAT)' > $@

FORCE:

$(BUILD_DIR)/%.o: src/%.cu $(CONFIG) Makefile
	@mkdir -p $(dir $@)
	$(COMPILE) $(SOURCE_LANG) -c $< -o $@

$(BUILD_DIR)/tests/%.o: tests/%.cu $(CONFIG) Makefile
	@mkdir -p $(dir $@)
	$(COMPILE) -Itests $(SOURCE_LANG) -c $< -o $@

-include $(OBJECTS:.o=.d) $(TEST_OBJECTS:.o=.d)

clean:
	rm -rf $(BUILD_DIR) $(EXEC)
