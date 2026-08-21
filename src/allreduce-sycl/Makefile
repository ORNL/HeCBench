#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
#MPI_ROOT  = /usr/lib/x86_64-linux-gnu/openmpi
#LAUNCHER  = /usr/bin/mpirun -n 1
# when CUDA enabled
MPI_ROOT  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/openmpi-4.1.5/
LAUNCHER  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/mpi/bin/mpirun -n 2

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp collectives.cpp timer.cpp

obj = $(source:.cpp=.o)

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(MPI_ROOT)/include -I. -DOMPI_SKIP_MPICXX= \
           -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -DOMPI_SKIP_MPICXX= 

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

%.o: %.cpp collectives.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
