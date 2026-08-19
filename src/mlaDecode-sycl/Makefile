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

# Backends to build (both on by default). Set MKL=no or DNN=no to disable one.
MKL       = yes
DNN       = yes

# path to the install of the oneMKL interface (only needed for CUDA/HIP targets)
ONEMKL_PATH ?= /path/to/oneMKL
# path to the oneDNN (DNNL) install
DNNL_PATH   ?= /opt/intel/oneapi/dnnl/latest
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -I../mlaDecode-cuda \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

LDFLAGS =

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

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

# ---- oneMKL backend ----
ifeq ($(MKL), yes)
  CFLAGS  += -DUSE_ONEMKL
  ifeq ($(ONEMKL), yes)
    CFLAGS  += -I$(ONEMKL_PATH)/include
  else
    CFLAGS  += -qmkl=parallel
    LDFLAGS += -qmkl=parallel
  endif
endif

# ---- oneDNN backend ----
ifeq ($(DNN), yes)
  CFLAGS  += -DUSE_ONEDNN -I$(DNNL_PATH)/include
  LDFLAGS += -L$(DNNL_PATH)/lib -ldnnl
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
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../mlaDecode-cuda/reference.h softmax.h kernel.h kernel_dnn.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 100
