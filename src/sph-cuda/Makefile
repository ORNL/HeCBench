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

build: $(program)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
# Fluid particle count (boundary count scales with it through the box spacing)
NUM_PARTICLES ?= 2048
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)
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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): fileio.o fluid.o geometry.o
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

fileio.o : fileio.cu particles.stamp
	$(CC) $(CFLAGS) -c $< -o $@

fluid.o : fluid.cu particles.stamp
	$(CC) $(CFLAGS) -c $< -o $@

geometry.o : geometry.cu particles.stamp
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
