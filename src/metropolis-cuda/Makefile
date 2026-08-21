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

obj = main.o utils.o heap.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

heap.o: heap.cu heap.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.cu heap.h utils.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cu utils.h kernel_metropolis.h kernel_prng.h kernel_reduction.h  
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.dat

dbg: $(program)
	./$(program) -l 32 11 -t 4.7 0.1 -a 1 1 1 1 -h 1.0 -z 7919

run: $(program)
	$(LAUNCHER) ./$(program) -l 32 11 -t 4.7 0.1 -a 10 2 2000 10 -h 1.0 -z 7919
