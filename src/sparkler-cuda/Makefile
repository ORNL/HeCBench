#===============================================================================
# User Options
#===============================================================================
NVHPC_SDK   ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5
MPI_ROOT    ?= $(NVHPC_SDK)/comm_libs/mpi
CUBLAS_ROOT ?= $(NVHPC_SDK)/math_libs/12.4
CUDART_ROOT ?= $(NVHPC_SDK)/cuda/12.4

# Compiler can be set below, or via environment variable
CC        = nvcc
LD        = mpiCC #$(MPI_ROOT)/bin/mpiCC
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?= mpirun -n 4

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
CFLAGS := $(EXTRA_CFLAGS) -I$(MPI_ROOT)/include -std=c++11 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = -L$(CUBLAS_ROOT)/targets/x86_64-linux/lib/ -lcublas \
	  -L$(CUDART_ROOT)/targets/x86_64-linux/lib/ -lcudart

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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
	$(LD) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --num_vector 4000 --num_field 90000 --num_iterations 10
