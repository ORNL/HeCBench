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

build: $(program)


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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): fileio.o fluid.o geometry.o
	$(CC) $(CFLAGS) $+ -o $@ $(LDFLAGS)

fileio.o : fileio.cu
	$(CC) $(CFLAGS) -c $< -o $@

fluid.o : fluid.cu  
	$(CC) $(CFLAGS) -c $< -o $@

geometry.o : geometry.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) fileio.o fluid.o geometry.o *.csv

run: $(program)
	$(LAUNCHER) ./$(program)
