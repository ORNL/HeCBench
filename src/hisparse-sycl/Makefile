#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp hisparse_kernels.cpp

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -I../hisparse-cuda \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

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

# One compile-and-link step: with separately compiled objects, DPC++'s
# amdgcn device link drops the kernel info and every launch asserts.
$(program): $(source) hisparse_kernels.h ../hisparse-cuda/reference.h Makefile
	$(CC) $(CFLAGS) $(source) -o $@ $(LDFLAGS)

.PHONY: clean run

clean:
	rm -rf $(program)

run: $(program)
	$(LAUNCHER) ./$(program) 8 256 100
	$(LAUNCHER) ./$(program) 32 1024 100
	$(LAUNCHER) ./$(program) 1 2048 100
	$(LAUNCHER) ./$(program) 64 2048 100
	$(LAUNCHER) ./$(program) 128 2048 100
	$(LAUNCHER) ./$(program) 256 2048 100
