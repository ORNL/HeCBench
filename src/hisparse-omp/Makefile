#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

# LLVM/AMD offload instead of icpx, e.g.
#   make CC=amdclang++ OFFLOAD="-fopenmp --offload-arch=gfx90a"
#   make CC=clang++ OFFLOAD="-fopenmp --offload-arch=sm_90"
OFFLOAD   =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp hisparse_kernels.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../hisparse-cuda

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifneq ($(OFFLOAD),)
  CFLAGS += $(OFFLOAD)
else ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp hisparse_kernels.h ../hisparse-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

hisparse_kernels.o: hisparse_kernels.cpp hisparse_kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean run

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 8 256 100
	$(LAUNCHER) ./$(program) 32 1024 100
	$(LAUNCHER) ./$(program) 1 2048 100
	$(LAUNCHER) ./$(program) 64 2048 100
	$(LAUNCHER) ./$(program) 128 2048 100
	$(LAUNCHER) ./$(program) 256 2048 100
