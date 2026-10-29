#===============================================================================
# User Options
#===============================================================================

CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_80
HIP       = no
HIP_ARCH  ?= gfx90a

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
source = main.cpp
obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -I$(MPI_ROOT)/include -I../moe-combine-cuda
LDFLAGS = -L$(MPI_ROOT)/lib -l$(MPI_LIB)

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

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

%.o: %.cpp combine.hpp ../moe-combine-cuda/reference.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --tokens 128 --hidden 1024 --iters 100 --warmup 100

.PHONY: clean run
