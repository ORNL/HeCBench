#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
# The kernels require the sm_90a feature set
ARCH      = -gencode arch=compute_90a,code=sm_90a
MIN_ARCH  = 900
LAUNCHER  ?=

KERNEL_DIR := twell_modules

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu \
	 $(KERNEL_DIR)/matmul_d2t.cu \
	 $(KERNEL_DIR)/matmul_t2d.cu \
	 $(KERNEL_DIR)/matmul_gated_t2d.cu \
	 $(KERNEL_DIR)/mlp_twell.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall $(ARCH) \
	  -DTWELL_MIN_ARCH=$(MIN_ARCH) \
	  --expt-relaxed-constexpr \
	  --expt-extended-lambda

# CUDA toolkit root (derived from nvcc location); used for the driver stub lib
CUDA_HOME ?= $(patsubst %/bin/,%,$(dir $(shell which $(CC))))

# Linker Flags
LDFLAGS = -L$(CUDA_HOME)/lib64/stubs -lcuda -lcublas 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 --use_fast_math
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu reference.cuh $(KERNEL_DIR)/torch_compat.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --reps 100
