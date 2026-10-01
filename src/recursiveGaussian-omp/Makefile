#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp RecursiveGaussianHost.cpp  shrUtils.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          -I../boxfilter-sycl -I../recursiveGaussian-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp ../boxfilter-sycl/shrUtils.h ../recursiveGaussian-cuda/main.h
	$(CC) $(CFLAGS) -c $< -o $@

RecursiveGaussianHost.o: ../recursiveGaussian-cuda/RecursiveGaussianHost.cpp \
                         ../recursiveGaussian-cuda/main.h 
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o : ../boxfilter-sycl/shrUtils.cpp ../boxfilter-sycl/shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../recursiveGaussian-cuda/StoneRGB.ppm 1000

