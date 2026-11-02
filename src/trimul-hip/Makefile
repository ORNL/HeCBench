#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fopenmp -I../trimul-cuda

# Linker Flags
LDFLAGS = -lhipblas -fopenmp

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
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

%.o: %.cu ../trimul-cuda/reference.h Makefile
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
