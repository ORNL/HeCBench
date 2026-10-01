#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
GPU       = yes
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = main.o

headers = \
	src/cuBool_gpu.cuh \
	src/bit_vector_kernels.cuh \
	src/float_kernels.cuh \
	src/cuBool_cpu.h \
	src/bit_vector_functions.h \
	src/config.h \
	src/io_and_allocation.hpp \
	src/updates_and_measures.cuh \
	src/helper/clipp.h \
	src/helper/rngpu.hpp \
	src/helper/cuda_helpers.cuh

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17

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

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU -Wall -DWARPSPERBLOCK=8 -fopenmp 
else
  CFLAGS += -x c++ -Wall -fopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: src/main.cu $(headers) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) data/*.data data/*.txt

# The dataset are available at https://github.com/Funatiq/cuBool 
run: $(program)
	$(LAUNCHER) ./$(program) ../bmf-cuda/data/MNIST.in -r 8 -l 112 -i 10000 --seed 123 --stuck 10000
