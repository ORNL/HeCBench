#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=
FIXTURE   ?= ../data/hydragnn-sc26/mace_pipeline_capture.macepipe

CUDA      = no
CUDA_ARCH ?= sm_52
HIP       = no
HIP_ARCH  ?= gfx90a
ONEMKL    = yes
MKLROOT   ?= /opt/intel/oneapi/mkl/latest

# oneMath is the renamed successor of the open-source oneMKL interfaces, and it
# is an alternative to ONEMKL rather than an addition. ONEMATH_PATH is an install
# prefix holding include/ and lib/.
ONEMATH      = no
ONEMATH_PATH ?= /path/to/oneMath

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl

# Linker Flags
LDFLAGS =
LDLIBS =

ifeq ($(ONEMATH),yes)
  CFLAGS += -DUSE_ONEMATH -I$(ONEMATH_PATH)/include
  LDFLAGS += -L$(ONEMATH_PATH)/lib -Wl,-rpath,$(ONEMATH_PATH)/lib
  ifeq ($(CUDA),yes)
    LDLIBS += -lonemath -lonemath_blas_cublas
  else ifeq ($(HIP),yes)
    LDLIBS += -lonemath -lonemath_blas_rocblas
  else
    LDLIBS += -lonemath
  endif
else ifeq ($(ONEMKL),yes)
  CFLAGS += -DUSE_ONEMKL -I$(MKLROOT)/include
  LDFLAGS += -L$(MKLROOT)/lib -Wl,-rpath,$(MKLROOT)/lib
  LDLIBS += -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core \
            -lpthread -lm -ldl
endif

ifeq ($(HIP),yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif
ifeq ($(CUDA),yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend=nvptx64-nvidia-cuda \
            --offload-arch=$(CUDA_ARCH)
endif
ifeq ($(DEVICE),cpu)
  CFLAGS += -DUSE_CPU
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -O0 -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS) $(LDLIBS)

%.o: %.cpp mace_pipeline.hpp mace_nlist.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

# The timed workload is a synthetic batch at the published SC26 dimensions:
# 128 graphs, 2726 atoms, 45974 edges. Correctness is validated on the captured
# fixture first.
run: $(program)
	$(LAUNCHER) ./$(program) --fixture $(FIXTURE) --warmup 100 --repeat 100

clean:
	rm -rf $(program) $(obj)
