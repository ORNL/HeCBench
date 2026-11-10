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

build: $(program)


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
# Fluid particle count (boundary count scales with it through the box spacing)
NUM_PARTICLES ?= 2048
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../include
CFLAGS += -DNUM_FLUID_PARTICLES=$(NUM_PARTICLES)

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

$(program): fileio.o fluid.o geometry.o
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

fileio.o : fileio.cpp particles.stamp
	$(CC) $(CFLAGS) -c $< -o $@

fluid.o : fluid.cpp particles.stamp
	$(CC) $(CFLAGS) -c $< -o $@

geometry.o : geometry.cpp particles.stamp
	$(CC) $(CFLAGS) -c $< -o $@

particles.stamp: FORCE
	@echo $(NUM_PARTICLES) | cmp -s - $@ || echo $(NUM_PARTICLES) > $@

FORCE:

clean:
	rm -rf $(program) fileio.o fluid.o geometry.o particles.stamp *.csv

run: $(program)
	$(LAUNCHER) ./$(program)

# Production problem size: 8x the fluid particles (about 64x the pair interactions)
PRODUCTION_PARTICLES ?= 16384

production:
	$(MAKE) -f $(firstword $(MAKEFILE_LIST)) NUM_PARTICLES=$(PRODUCTION_PARTICLES)
