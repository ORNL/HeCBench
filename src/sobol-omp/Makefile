#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = sobol.cpp sobol_gold.cpp sobol_gpu.cpp sobol_primitives.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall 

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

$(program): $(obj) 
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

sobol.o:sobol.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol_gold.o:sobol_gold.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol_gpu.o:sobol_gpu.cpp
	$(CC) $(CFLAGS) -c $< -o $@

sobol_primitives.o:sobol_primitives.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1000000 1000 100
