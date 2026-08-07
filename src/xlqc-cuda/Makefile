#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall $(GSL_INC) -arch=$(ARCH)

# Linker Flags
LDFLAGS = $(GSL_LIB)

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
	$(CC) -dc -o $@ $< $(CFLAGS)

scf.o: scf.cc
	$(CC) -c -o $@ $< $(CFLAGS)

basis.o: basis.cc
	$(CC) -c -o $@ $< $(CFLAGS)

crys.o: ./int_lib/crys.cc
	$(CC) -c -o $@ $< $(CFLAGS)

cints.o: ./int_lib/cints.cc
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) sp
	$(LAUNCHER) ./$(program) dp
