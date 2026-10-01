#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = amdclang++
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
ARCH     ?= gfx906
ROCM_PATH ?= /opt/rocm
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = saxpy.c hsaxpy.c asaxpy.c check1ns.c wtcalc.c

obj = $(source:.c=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -Wall -D__HIP_PLATFORM_AMD__ \
          -I$(ROCM_PATH)/include -I$(ROCM_PATH)/include/hipblas

# Linker Flags
LDFLAGS = -L$(ROCM_PATH)/lib -lhipblas -lamdhip64

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -target x86_64-pc-linux-gnu \
          -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa \
          -Xopenmp-target=amdgcn-amd-amdhsa \
          -march=$(ARCH)
else
  CFLAGS +=
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.c asaxpy.h check1ns.h hsaxpy.h wtcalc.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
