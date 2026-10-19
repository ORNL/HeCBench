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

NX     ?= 128
NY     ?= 128
NZ     ?= 128
NODES  ?= 512
WARMUP ?= 1000
ITERS  ?= 1000

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../condor-cuda -fsycl

ifneq ($(GCC_TOOLCHAIN),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

LDFLAGS =

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

ifeq ($(GPU),yes)
  CFLAGS += -DUSE_GPU
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../condor-cuda/condor_reference.hpp ../condor-cuda/condor_host.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --nx $(NX) --ny $(NY) --nz $(NZ) \
        --nodes $(NODES) --iters $(ITERS) --warmup $(WARMUP)

.PHONY: clean run
