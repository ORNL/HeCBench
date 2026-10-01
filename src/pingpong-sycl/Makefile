#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
MPI_ROOT  ?= /opt/intel/oneapi/2024.1
LAUNCHER  ?= $(MPI_ROOT)/bin/mpirun -n 2

# when CUDA enabled for the MPI pingpong
#NVHPC_SDK = /opt/nvidia/hpc_sdk/Linux_x86_64/24.5
#MPI_ROOT  = $(NVHPC_SDK)/comm_libs/12.4/openmpi4/openmpi-4.1.5/
#LAUNCHER  = $(NVHPC_SDK)/comm_libs/mpi/bin/mpirun --mca coll ^hcoll -n 2

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"


#===============================================================================
# Program name & source code list
#===============================================================================

mpi-program = main-mpi
ccl-program = main-ccl

mpi-source = main-mpi.cpp
ccl-source = main-ccl.cpp

mpi-obj = $(mpi-source:.cpp=.o)
ccl-obj = $(ccl-source:.cpp=.o)

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(MPI_ROOT)/include \
           -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -lccl

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

$(mpi-program): $(mpi-obj)
	$(CC) $(CFLAGS) $(mpi-obj) -o $@ $(LDFLAGS)

$(ccl-program): $(ccl-obj)
	$(CC) $(CFLAGS) $(ccl-obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(mpi-program) $(mpi-obj)
	rm -rf $(ccl-program) $(ccl-obj)

run: $(mpi-program) $(ccl-program)
	$(LAUNCHER) ./$(mpi-program)
	$(LAUNCHER) ./$(ccl-program)
