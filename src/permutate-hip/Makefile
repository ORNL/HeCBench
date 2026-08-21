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

source = ../permutate-cuda/main.cpp \
         ../permutate-cuda/permutation_testing.cpp \
         ../permutate-cuda/utils.cpp \
         ../permutate-cuda/statistical_test.cpp \
         ../permutate-cuda/bzip/blocksort.cpp \
         ../permutate-cuda/bzip/bzlib.cpp \
         ../permutate-cuda/bzip/compress.cpp \
         ../permutate-cuda/bzip/crctable.cpp \
         ../permutate-cuda/bzip/decompress.cpp \
         ../permutate-cuda/bzip/huffman.cpp \
         ../permutate-cuda/bzip/randtable.cpp

kernels = gpu_permutation_testing.cu

obj = main.o permutation_testing.o utils.o statistical_test.o \
      blocksort.o bzlib.o compress.o crctable.o decompress.o \
      huffman.o randtable.o gpu_permutation_testing.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++17 -Wall -I../permutate-cuda

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
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu kernel_functions.cuh device_functions.cuh \
          ../permutate-cuda/header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: ../permutate-cuda/%.cpp ../permutate-cuda/header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: ../permutate-cuda/bzip/%.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../permutate-cuda/test_data/truerand_1bit.bin
