#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = mpiicpc
CXX       = icpx
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?= mpirun -n 4

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
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
CFLAGS := $(EXTRA_CFLAGS) -cxx=${CXX} -std=c++17 -Wall \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -fsycl -I${MKLROOT}/include -DMKL_ILP64 

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -fsycl ${MKLROOT}/lib/intel64/libmkl_sycl.a \
          -fsycl-device-code-split=per_kernel \
          -Wl,-export-dynamic -Wl,--start-group \
          ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a \
          ${MKLROOT}/lib/intel64/libmkl_tbb_thread.a \
          ${MKLROOT}/lib/intel64/libmkl_core.a \
          -Wl,--end-group -ltbb -lsycl -lOpenCL -lpthread -lm -ldl 

ifeq ($(VENDOR), AdaptiveCpp)
    LDFLAGS_TMP := $(LDFLAGS)
    LDFLAGS = $(filter-out -fsycl, $(LDFLAGS_TMP))
endif

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
	$(LAUNCHER) ./$(program) --num_vector 4000 --num_field 90000 --num_iterations 10
