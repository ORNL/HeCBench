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

source = main.cpp warpsort.cu

obj = main.o warpsort.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DWARP_SIZE=32 -DHALF_WARP_SIZE=16 \
          -std=c++17 -I. -Wall

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

main.o: main.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

warpsort.o: warpsort.cu cuda/CudaUtils.cuh \
                        cuda/Comparators.cuh \
                        cuda/DeviceTensor.cuh \
                        cuda/DeviceTensor-inl.cuh \
                        cuda/NumericLimits.cuh \
                        cuda/Pair.cuh \
                        cuda/RegisterUtils.cuh \
                        cuda/ShuffleTypes.cuh \
                        cuda/WarpBitonicSort.cuh \
                        Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1000
