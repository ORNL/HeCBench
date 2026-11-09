#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER ?=
# Target GPU, for example gfx90a or gfx942. Empty leaves the choice to hipcc.
HIP_ARCH ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu forward.cu baseline.cu optimized.cu

obj = $(source:.cu=.o) reference.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../gsplat3d-cuda

ifneq ($(HIP_ARCH),)
  CFLAGS += --offload-arch=$(HIP_ARCH)
endif

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu launch.h warp_sum.cuh ../gsplat3d-cuda/types.h ../gsplat3d-cuda/reference.h ../gsplat3d-cuda/raster_math.h Makefile
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
