#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
ONEMKL    = no
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

# path to the install of the oneMKL interface
ONEMKL_PATH   ?= /path/to/oneMKL

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags (OpenMP parallelizes the CPU reference/verifier)
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -fopenmp \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -fopenmp

ifeq ($(CUDA), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
  LDFLAGS += -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_cublas
endif

ifeq ($(HIP), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
  LDFLAGS += -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_rocblas
endif

ifeq ($(ONEMKL), yes)
  CFLAGS += -I$(ONEMKL_PATH)/include
else
  CFLAGS += -qmkl=parallel
  LDFLAGS += -qmkl=parallel
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

%.o: %.cpp ../muon-cuda/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 3072 768 10 1
	$(LAUNCHER) ./$(program) 768 3072 10 1
	$(LAUNCHER) ./$(program) 3072 768 10000 0
	$(LAUNCHER) ./$(program) 768 3072 10000 0
