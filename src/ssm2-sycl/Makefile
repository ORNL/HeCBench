#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

# Mamba-2 2.7B SSD layer (state-spaces/mamba2-2.7b):
#   d_model=2560, expand=2, headdim=64 -> 80 heads
#   ngroups=1, d_state=128, 2K prefill.
# The benchmark sweeps chunk sizes 32, 64, and 128.
REPEAT ?= 100
BATCH  ?= 8
TOKENS ?= 2048
HEADS  ?= 80
GROUPS ?= 1
HEAD   ?= 64
STATE  ?= 128

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_80
HIP       = no
HIP_ARCH  ?= gfx90a
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

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -Xarch_host -fopenmp -fsycl \
          -I../ssm2-cuda

ifneq ($(strip $(GCC_TOOLCHAIN)),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Link OpenMP through the compiler so icpx selects its runtime. Keeping the
# option host-scoped also prevents it from being sent to CUDA or HIP devices.
LDFLAGS = -Xarch_host -fopenmp

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

%.o: %.cpp kernels.h ../ssm2-cuda/reference.h \
          ../ssm2-cuda/ssm2_launch_policy.h ../ssm2-cuda/ssm2_types.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(REPEAT) $(BATCH) $(TOKENS) $(HEADS) $(GROUPS) $(HEAD) $(STATE)

.PHONY: clean run
