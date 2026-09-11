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
# BF16 joint_matrix requires Ampere or newer.
CUDA_ARCH ?= sm_80
HIP       = no
HIP_ARCH  ?= gfx90a
ONEMKL    = no
# path to the install of the oneMKL interface
ONEMKL_PATH ?= /path/to/oneMKL
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../trimul-cuda -fsycl

ifneq ($(strip $(GCC_TOOLCHAIN)),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
    CFLAGS += -fopenmp
    LDFLAGS = -fopenmp
else
    # -Xarch_host keeps OpenMP on the host compilation for DPC++
    CFLAGS += -Xarch_host -fopenmp
    LDFLAGS = -Xarch_host -fopenmp
endif

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

ifeq ($(ONEMKL), yes)
  CFLAGS += -DUSE_ONEMKL
  ifeq ($(CUDA), yes)
    CFLAGS  += -I$(ONEMKL_PATH)/include
    LDFLAGS += -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_cublas
  else ifeq ($(HIP), yes)
    CFLAGS  += -I$(ONEMKL_PATH)/include
    LDFLAGS += -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_rocblas
  else
    CFLAGS  += -qmkl=parallel
    LDFLAGS += -qmkl=parallel -lmkl_sycl_blas
  endif
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

%.o: %.cpp ../trimul-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1 256 128 100
	$(LAUNCHER) ./$(program) 1 384 128 100
	$(LAUNCHER) ./$(program) 1 640 128 100
	$(LAUNCHER) ./$(program) 1 768 128 100
