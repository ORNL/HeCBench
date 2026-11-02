#===============================================================================
# User Options
#===============================================================================

CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

# A loaded Cray MPICH module sets these to that version's prefix, including
# installs that are not 9.0.1 and not under /opt. Otherwise use OpenMPI.
CRAY_MPI_DIR := $(or $(CRAY_MPICH_DIR),$(MPICH_DIR))
ifneq ($(wildcard $(CRAY_MPI_DIR)/include/mpi.h),)
  MPI_ROOT ?= $(CRAY_MPI_DIR)
  MPI_LIB  ?= mpi_cray
else
  MPI_ROOT ?= /usr/lib/x86_64-linux-gnu/openmpi
  MPI_LIB  ?= mpi
endif

#===============================================================================
# Program name & source code list
#===============================================================================

program = main
source = main.cu
obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(MPI_ROOT)/include -Wall
LDFLAGS = -L$(MPI_ROOT)/lib -l$(MPI_LIB)

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../moe-combine-cuda/combine.cuh ../moe-combine-cuda/reference.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --tokens 128 --hidden 1024 --iters 100 --warmup 100

.PHONY: clean run
