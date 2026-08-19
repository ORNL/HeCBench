ROCM_PATH ?= /opt/rocm
#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

HIP_ARCH  ?= gfx90a
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu shrUtils.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
#--hipstdpar-path=/opt/rocm/include/thrust/system/hip/hipstdpar

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          -I$(ROCM_PATH)/include/hipblas \
          --hipstdpar \
          --offload-arch=$(HIP_ARCH)


# Linker Flags
LDFLAGS = -lhipblas

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -munsafe-fp-atomics
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 268435456 1000
