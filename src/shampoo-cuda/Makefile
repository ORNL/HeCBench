#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall,-fopenmp -arch=$(ARCH)

# Linker Flags
LDFLAGS = -Xcompiler -fopenmp -lcublas -lcusolver

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

verify: $(program)
	$(LAUNCHER) ./$(program) 33 31 3 1 both
	$(LAUNCHER) ./$(program) 1024 1024 1 1 both

perf: verify
	$(LAUNCHER) ./$(program) 4096 4096 2 0 both
	$(LAUNCHER) ./$(program) 8192 8192 2 0 both

run: perf

.PHONY: clean run verify perf
