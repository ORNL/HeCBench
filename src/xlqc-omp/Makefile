#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
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

obj=basis.o scf.o main.o crys.o cints.o


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../xlqc-cuda -I../xlqc-cuda/int_lib $(GSL_INC)

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

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CFLAGS)

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
