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

build: $(program)


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../include

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

fileio.o : fileio.cpp
	$(CC) $(CFLAGS) -c $< -o $@

fluid.o : fluid.cpp  
	$(CC) $(CFLAGS) -c $< -o $@

geometry.o : geometry.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) fileio.o fluid.o geometry.o *.csv

run: $(program)
	$(LAUNCHER) ./$(program)
