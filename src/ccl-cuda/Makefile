#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
#MPI_ROOT  = /usr/lib/x86_64-linux-gnu/openmpi
#LAUNCHER  = /usr/bin/mpirun -n 2
NVHPC_SDK ?= /opt/nvidia/hpc_sdk/Linux_x86_64/26.5
MPI_ROOT  ?= $(NVHPC_SDK)/comm_libs/13.2/hpcx/hpcx-2.50/ompi4
NCCL_ROOT ?= $(NVHPC_SDK)/comm_libs/nccl
LAUNCHER  ?= $(NVHPC_SDK)/comm_libs/mpi/bin/mpirun -n 2

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(MPI_ROOT)/include -I$(NCCL_ROOT)/include \
          -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -L$(NCCL_ROOT)/lib -lnccl

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 50
