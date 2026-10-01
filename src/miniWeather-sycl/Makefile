#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

ifeq ($(CUDA), yes)
  MPI_INCLUDE ?= "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/mpi/include"
  MPI_LIB_DIR ?= "/opt/nvidia/hpc_sdk/Linux_x86_64/22.11/comm_libs/mpi/lib"
else
  MPI_INCLUDE ?= "/usr/lib/x86_64-linux-gnu/openmpi/include"
  MPI_LIB_DIR ?= "/usr/lib/x86_64-linux-gnu/openmpi/lib"
endif

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) \
          -DNX=400 \
       	  -DNZ=200 \
	  -DSIM_TIME=600 \
	  -DDATA_SPEC=DATA_SPEC_THERMAL \
          -I$(MPI_INCLUDE) \
          -std=c++17 -Wall -fsycl \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -I../miniWeather-cuda

# Linker Flags
LDFLAGS = -L$(MPI_LIB_DIR) -lmpi

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../miniWeather-cuda/check_output.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) mpiexec -n 1 ./$(program)
