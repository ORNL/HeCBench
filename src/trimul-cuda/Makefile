#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
# BF16 WMMA requires Ampere or newer.
ARCH      = sm_80
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

# Standard Flags
# The CPU reference is OpenMP-parallel; without -fopenmp it still builds and
# runs, just serially, which is slow at the larger sequence lengths.
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -Xcompiler -fopenmp \
          -arch=$(ARCH)

# Linker Flags
LDFLAGS = -lcublas -Xcompiler -fopenmp

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

%.o: %.cu reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Sequence lengths AlphaFold actually runs at, with the pair-representation
# width of 128: the AF2 training crops (256, 384) and the AF3 ones (384, 640,
# 768).  Batch is 1 per device in both.
run: $(program)
	$(LAUNCHER) ./$(program) 1 256 128 100
	$(LAUNCHER) ./$(program) 1 384 128 100
	$(LAUNCHER) ./$(program) 1 640 128 100
	$(LAUNCHER) ./$(program) 1 768 128 100
