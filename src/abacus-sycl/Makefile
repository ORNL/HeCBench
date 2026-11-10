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

# One process runs the Si16, Pt, and Fe spherical-harmonic sizes, then the
# neighbor list. See README.md. ITERS is the timed launch count.
ITERS ?= 100

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = ../abacus-cuda/main.cpp ylm.cpp neighborlist.cpp ../abacus-cuda/reference.cpp

obj = main.o ylm.o neighborlist.o reference.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -I../abacus-cuda

ifneq ($(GCC_TOOLCHAIN),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS =

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
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS += -DUSE_GPU
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../abacus-cuda/main.cpp ../abacus-cuda/kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

ylm.o: ylm.cpp ../abacus-cuda/kernels.h ../abacus-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

neighborlist.o: neighborlist.cpp ../abacus-cuda/kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: ../abacus-cuda/reference.cpp ../abacus-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(ITERS)

.PHONY: clean run
