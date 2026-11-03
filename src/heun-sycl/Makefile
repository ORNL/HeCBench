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

ONEDNN_PATH ?= /opt/intel/oneapi/dnnl/latest
TBB_PATH    ?= /opt/intel/oneapi/tbb/latest
ICX_LIB     ?= /opt/intel/oneapi/compiler/latest/lib

# Production sampler shape: one forecast, 99 HRRR variables, 512 x 640.
BATCH     ?= 1
CHANNELS  ?= 99
HEIGHT    ?= 512
WIDTH     ?= 640
STEPS     ?= 18

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -pthread -fsycl \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -pthread

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

CFLAGS += -I$(ONEDNN_PATH)/include
LDFLAGS += -L$(ONEDNN_PATH)/lib -L$(TBB_PATH)/lib -L$(ICX_LIB) \
           -Wl,-rpath,$(ONEDNN_PATH)/lib -Wl,-rpath,$(TBB_PATH)/lib \
           -Wl,-rpath,$(ICX_LIB) -ldnnl -ltbb -lsvml -limf -lintlc

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g
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

all: $(program)

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../heun-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: all
	$(LAUNCHER) ./$(program) 100 $(BATCH) $(CHANNELS) $(HEIGHT) $(WIDTH) $(STEPS)
