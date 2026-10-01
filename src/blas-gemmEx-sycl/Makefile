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
ONEMKL    = no
ONEMATH      = no
ONEMATH_PATH ?= /path/to/oneMath
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

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -I../blas-gemmEx-cuda \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

ifeq ($(CUDA), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
  ifeq ($(ONEMATH), yes)
    LDFLAGS = -L$(ONEMATH_PATH)/lib -lonemath -lonemath_blas_cublas
  else
    LDFLAGS = -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_cublas
  endif
else ifeq ($(HIP), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
  ifeq ($(ONEMATH), yes)
    LDFLAGS = -L$(ONEMATH_PATH)/lib -lonemath -lonemath_blas_rocblas
  else
    LDFLAGS = -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_rocblas
  endif
else ifeq ($(ONEMATH), yes)
  LDFLAGS = -L$(ONEMATH_PATH)/lib -lonemath
endif

ifeq ($(ONEMATH), yes)
  CFLAGS += -I$(ONEMATH_PATH)/include -DUSE_ONEMATH
else ifeq ($(ONEMKL), yes)  
  CFLAGS += -I$(ONEMKL_PATH)/include
else
  CFLAGS += -qmkl=parallel
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

%.o: %.cpp ../blas-gemmEx-cuda/utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 4096 4096 4096 100
	$(LAUNCHER) ./$(program) 8192 8192 8192 100
