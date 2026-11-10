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
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp forward.cpp baseline.cpp optimized.cpp

obj = $(source:.cpp=.o) reference.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../gsplat3d-cuda \
          -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS =

# Each backward kernel is built for sub-group sizes 32 and 64, and the one the
# device supports is chosen at run time. NVIDIA has no 64-lane sub-groups and
# AMD CDNA has no 32-lane ones, so the compiler warns about the other size.
# The sources are compiled separately and linked. On AMD the old offload
# driver links a binary whose kernels the runtime cannot find.
ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH) \
            -Wno-incorrect-sub-group-size --offload-new-driver
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) \
	    -Wno-incorrect-sub-group-size --offload-new-driver
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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp launch.h warp_sum.h ../gsplat3d-cuda/types.h ../gsplat3d-cuda/reference.h ../gsplat3d-cuda/raster_math.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: ../gsplat3d-cuda/reference.cpp ../gsplat3d-cuda/reference.h ../gsplat3d-cuda/types.h ../gsplat3d-cuda/raster_math.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# The four cases from the gsplat backward rasterizer study.
# Arguments are gaussians, height, width, and timed iterations.
run: $(program)
	$(LAUNCHER) ./$(program) 4096 128 128 100
	$(LAUNCHER) ./$(program) 32768 256 256 100
	$(LAUNCHER) ./$(program) 131072 512 384 100
	$(LAUNCHER) ./$(program) 524288 648 420 100
