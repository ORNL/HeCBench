#===============================================================================
# User Options
#===============================================================================

CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl \
          -I../rmsnorm-rope-cuda \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

LDFLAGS = $(EXTRA_LDFLAGS)

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=$(HIP_ARCH)
endif

ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(source) ../rmsnorm-rope-cuda/reference.h
	$(CC) $(CFLAGS) $(source) -o $@ $(LDFLAGS)

%.o: %.cpp ../rmsnorm-rope-cuda/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 4096 18 256 24 20 20 100
	$(LAUNCHER) ./$(program) 4096 18 256 22 21 21 100
	$(LAUNCHER) ./$(program) 16384 18 256 24 20 20 100
	$(LAUNCHER) ./$(program) 16384 18 256 22 21 21 100

test: $(program)
	$(LAUNCHER) ./$(program) 1000 3 64 8 4 4 100
	$(LAUNCHER) ./$(program) 999 5 128 16 8 8 100
