#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no

SYCL      = yes
GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_90
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/packages/gcc/10.2.0/"

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	#$(LAUNCHER) ./$(program) 0 16 16 16 100 1
	$(LAUNCHER) ./$(program) 0 16 16 64 100 1
	$(LAUNCHER) ./$(program) 0 32 32 64 100 1
	$(LAUNCHER) ./$(program) 0 256 256 256 100 1
	$(LAUNCHER) ./$(program) 1 256 256 256 100 1
	$(LAUNCHER) ./$(program) 0 512 512 512 100 1
	$(LAUNCHER) ./$(program) 1 512 512 512 100 1
	$(LAUNCHER) ./$(program) 0 1024 1024 1024 100 0
	$(LAUNCHER) ./$(program) 1 1024 1024 1024 100 0
	$(LAUNCHER) ./$(program) 0 4096 4096 4096 100 0
	$(LAUNCHER) ./$(program) 1 4096 4096 4096 100 0
	$(LAUNCHER) ./$(program) 0 8192 8192 8192 100 0
	$(LAUNCHER) ./$(program) 1 8192 8192 8192 100 0
