#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
MPI_INCLUDE ?= "/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_LIB_DIR ?= "/usr/lib/x86_64-linux-gnu/openmpi/lib"
LAUNCHER    ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -DNX=400 \
       	  -DNZ=200 \
	  -DSIM_TIME=600 \
	  -DDATA_SPEC=DATA_SPEC_THERMAL \
          -I$(MPI_INCLUDE) \
          -std=c++17 -Wall \
          -I../miniWeather-cuda

# Linker Flags
LDFLAGS = -L$(MPI_LIB_DIR) -lmpi

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
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

%.o: %.cu ../miniWeather-cuda/check_output.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) mpiexec -n 1 ./$(program)
