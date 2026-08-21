#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

# HIP device specific
UNSAFE_FP_ATOMICS = yes


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o utils.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../gd-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

ifeq ($(UNSAFE_FP_ATOMICS),yes)
  CFLAGS += -munsafe-fp-atomics -DUNSAFE_FP_ATOMICS
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cu ../gd-cuda/utils.h ../gd-cuda/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: ../gd-cuda/utils.cpp ../gd-cuda/utils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) gisette_scale 0.0001 10 100  
