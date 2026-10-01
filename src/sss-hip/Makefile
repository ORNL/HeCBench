#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../sss-cuda

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

SHARED_SRC = ../sss-cuda/DPmixGGM.cpp ../sss-cuda/DPmixGGM_Lists.cpp \
             ../sss-cuda/graph.cpp ../sss-cuda/gwish.cpp \
             ../sss-cuda/utilities.cpp ../sss-cuda/graph.h \
             ../sss-cuda/gsl_compat.h ../sss-cuda/kernels.cu

%.o: %.cu DPmixGGM_SSSmoves.cpp $(SHARED_SRC) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) RES/f9_n150_p50_modes_GPU.txt

run: $(program)
	$(LAUNCHER) ./$(program) f9_n150_p50
