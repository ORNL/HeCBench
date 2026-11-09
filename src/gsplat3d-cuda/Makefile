#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_75
LAUNCHER ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu forward.cu baseline.cu optimized.cu reference.cpp

cu_source = $(filter %.cu,$(source))
cpp_source = $(filter %.cpp,$(source))
obj = $(cu_source:.cu=.o) $(cpp_source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

%.o: %.cu launch.h types.h warp_sum.cuh reference.h raster_math.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp reference.h types.h raster_math.h Makefile
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
