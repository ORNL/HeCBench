#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
#MPI_ROOT  = /usr/lib/x86_64-linux-gnu/openmpi
#LAUNCHER  = /usr/bin/mpirun -n 1
MPI_ROOT  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/12.4/openmpi4/openmpi-4.1.5/
LAUNCHER  ?= /opt/nvidia/hpc_sdk/Linux_x86_64/24.5/comm_libs/mpi/bin/mpirun -n 2

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu collectives.cu timer.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(MPI_ROOT)/include -DOMPI_SKIP_MPICXX= \
          -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -DOMPI_SKIP_MPICXX=

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

%.o: %.cu *.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
