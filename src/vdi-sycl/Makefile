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
CUDA_ARCH ?= sm_90
HIP       = no
HIP_ARCH  ?= gfx90a
ONEMKL    = no
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

# path to the install of the oneMKL interface
ONEMKL_PATH ?= /path/to/oneMKL

# Representative workload from NVlabs/kda issue 6.
# `python vdn_delta_inverse_ref.py` with no arguments is exactly this shape:
#   101 frames x 7 heads = 707 matrices, d = 128, 1008 tokens per frame.
FRAMES ?= 101
HEADS  ?= 7
TOKENS ?= 1008
DIM    ?= 128
REPEAT ?= 50

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp fused.cpp libraries.cpp

obj = $(source:.cpp=.o)

headers = fused.h libraries.h ../vdi-cuda/host_common.hpp \
          ../vdi-cuda/reference.h ../vdi-cuda/local_layout.hpp

#===============================================================================
# Sets Flags
#===============================================================================

# fp32 Cholesky and the inverse products must not use fast-math sqrt/div approximations. 
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../vdi-cuda -fsycl -DWDN_ONEMKL

ifneq ($(strip $(GCC_TOOLCHAIN)),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -lpthread

ifeq ($(CUDA), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
  LDFLAGS += -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_cublas -lonemkl_lapack_cusolver
endif

ifeq ($(HIP), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
  LDFLAGS += -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_rocblas -lonemkl_lapack_rocsolver
endif

ifeq ($(ONEMKL), yes)
  CFLAGS += -I$(ONEMKL_PATH)/include
else
  CFLAGS += -qmkl=parallel
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
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp $(headers) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

.PHONY: clean run

run: $(program)
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 1
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 50
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 100
