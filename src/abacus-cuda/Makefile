#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?=

# One process runs the Si16, Pt, and Fe spherical-harmonic sizes, then the
# neighbor list. See README.md. ITERS is the timed launch count.
ITERS ?= 100

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp ylm.cu neighborlist.cu reference.cpp

obj = main.o ylm.o neighborlist.o reference.o

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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

ylm.o: ylm.cu kernels.h reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

neighborlist.o: neighborlist.cu kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: reference.cpp reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(ITERS)

.PHONY: clean run
