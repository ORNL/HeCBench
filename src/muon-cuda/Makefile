#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
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

# Standard Flags (OpenMP parallelizes the CPU reference/verifier)
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall,-fopenmp -arch=$(ARCH)

# Linker Flags
LDFLAGS = -Xcompiler -fopenmp -lcublas

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 #--use_fast_math
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 3072 768 10 1
	$(LAUNCHER) ./$(program) 768 3072 10 1
	$(LAUNCHER) ./$(program) 3072 768 10000 0
	$(LAUNCHER) ./$(program) 768 3072 10000 0
