#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

# One process runs the Si16, Pt, and Fe spherical-harmonic sizes, then the
# neighbor list. See README.md. ITERS is the timed launch count.
ITERS ?= 100

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = ../abacus-cuda/main.cpp ylm.cpp neighborlist.cpp ../abacus-cuda/reference.cpp

obj = main.o ylm.o neighborlist.o reference.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../abacus-cuda

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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: ../abacus-cuda/main.cpp ../abacus-cuda/kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

ylm.o: ylm.cpp ../abacus-cuda/kernels.h ../abacus-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

neighborlist.o: neighborlist.cpp ../abacus-cuda/kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: ../abacus-cuda/reference.cpp ../abacus-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(ITERS)

.PHONY: clean run
