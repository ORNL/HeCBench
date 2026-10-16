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
HIP_ARCH  ?= gfx90a
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

# Standard Flags (OpenMP parallelizes the CPU reference)
# -Xarch_host keeps OpenMP on the host. A bare -fopenmp is also applied to
# the NVIDIA or AMD device link and fails looking for the device OpenMP runtime.
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -Xarch_host -fopenmp -I../shampoo-cuda

# Empty --gcc-toolchain breaks this DPC++ when GCC_TOOLCHAIN is unset.
ifneq ($(strip $(GCC_TOOLCHAIN)),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl -Xarch_host, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -Xarch_host -fopenmp

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
  CFLAGS += -DUSE_GPU
endif

ifeq ($(VENDOR), AdaptiveCpp)
  LDFLAGS := $(filter-out -Xarch_host, $(LDFLAGS))
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../shampoo-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 33 31 100 1
	$(LAUNCHER) ./$(program) 1024 1024 1 1
	$(LAUNCHER) ./$(program) 8192 8192 2 0

.PHONY: clean run
