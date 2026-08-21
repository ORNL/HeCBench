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

source = main.cpp permutation_testing.cpp utils.cpp statistical_test.cpp \
         bzip/blocksort.cpp \
         bzip/bzlib.cpp \
         bzip/compress.cpp \
         bzip/crctable.cpp \
         bzip/decompress.cpp \
         bzip/huffman.cpp \
         bzip/randtable.cpp

kernels = gpu_permutation_testing.cu

obj = $(source:.cpp=.o) $(kernels:.cu=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

%.o: %.cu kernel_functions.cuh device_functions.cuh header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp Makefile header.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) test_data/truerand_1bit.bin
