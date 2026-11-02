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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl

ifneq ($(strip $(GCC_TOOLCHAIN)),)
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
	    -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$(HIP_ARCH)
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

# Compile and link together. Separate compilation drops this DPC++ AMD
# device image before the runtime can find its kernels.
$(program): $(source) ../index-shuffle-cuda/reference.h Makefile
	$(CC) $(CFLAGS) $(source) -o $@ $(LDFLAGS)

clean:
	rm -rf $(program) $(obj)

# Tokens in {4096, 32768}, experts in {16, 128}, top-k in {1, 2, 4}.
TOKENS  = 4096 32768
EXPERTS = 16 128
TOPKS   = 1 2 4

run: $(program)
	@set -e; \
	for t in $(TOKENS); do for e in $(EXPERTS); do for k in $(TOPKS); do \
	  $(LAUNCHER) ./$(program) $$t $$e $$k 1000; \
	done; done; done
