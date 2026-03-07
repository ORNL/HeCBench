#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = sobol.cu sobol_gold.cu sobol_gpu.cu sobol_primitives.cu

obj = $(source:.cu=.o)

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

$(program): $(obj) 
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

sobol.o:sobol.cu
	$(CC) $(CFLAGS) -c $< -o $@

sobol_gold.o:sobol_gold.cu
	$(CC) $(CFLAGS) -c $< -o $@

sobol_gpu.o:sobol_gpu.cu
	$(CC) $(CFLAGS) -c $< -o $@

sobol_primitives.o:sobol_primitives.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1000000 1000 100
