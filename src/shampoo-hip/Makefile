#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
ROCM_PATH ?= /opt/rocm
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags (OpenMP parallelizes the CPU reference)
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fopenmp -I$(ROCM_PATH)/include -I../shampoo-cuda

# Linker Flags
LDFLAGS = -fopenmp -L$(ROCM_PATH)/lib -lhipblas

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
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

%.o: %.cu ../shampoo-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 33 31 100 1
	$(LAUNCHER) ./$(program) 1024 1024 1 1
	$(LAUNCHER) ./$(program) 8192 8192 2 0

.PHONY: clean run
