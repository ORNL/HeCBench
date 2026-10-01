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
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

# path to the install of the oneMKL interface
ONEMKL_PATH   ?= /path/to/oneMKL

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp device.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -I../pcc-cuda \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -fno-sycl-id-queries-fit-in-int

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

ifeq ($(CUDA), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
  LDFLAGS = -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_cublas
endif

ifeq ($(HIP), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
  LDFLAGS = -L$(ONEMKL_PATH)/lib -lonemkl -lonemkl_blas_rocblas
endif

ifeq ($(ONEMKL), yes)
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

main.o: ../pcc-cuda/main.cpp ../pcc-cuda/device.h
	$(CC) $(CFLAGS) -c $< -o $@

device.o: device.cpp ../pcc-cuda/device.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) corrs.txt

run: $(program)
	$(LAUNCHER) ./$(program) 90112 165
