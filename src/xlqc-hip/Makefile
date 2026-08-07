#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
GSL_INC   ?= $(shell gsl-config --cflags)
GSL_LIB   ?= $(shell gsl-config --libs)
LAUNCHER  =

ifeq ($(GSL_LIB),)
$(error gsl-config not found; install libgsl-dev or set GSL_INC/GSL_LIB)
endif

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj=basis.o scf.o main.o crys.o cints.o cuda_rys_sp.o cuda_rys_dp.o


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../xlqc-cuda -I../xlqc-cuda/int_lib $(GSL_INC)

# Linker Flags
LDFLAGS = $(GSL_LIB) -fgpu-rdc --hip-link

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu cuda_rys_dp.h cuda_rys_sp.h
	$(CC) -fgpu-rdc -c -o $@ $< $(CFLAGS)

scf.o: ../xlqc-cuda/scf.cc
	$(CC) -c -o $@ $< $(CFLAGS)

basis.o: ../xlqc-cuda/basis.cc
	$(CC) -c -o $@ $< $(CFLAGS)

crys.o: ../xlqc-cuda/int_lib/crys.cc
	$(CC) -c -o $@ $< $(CFLAGS)

cints.o: ../xlqc-cuda/int_lib/cints.cc
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) sp
	$(LAUNCHER) ./$(program) dp
