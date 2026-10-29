#===============================================================================
# User Options
#===============================================================================

# Intel oneAPI OpenMP offload (repo default). AMD and NVIDIA builds are
# Makefile.aomp and Makefile.nvc.
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

# One zone per resident full block. Each launch runs STEPS backward-Euler
# steps. WARMUP is the untimed full launch count, ITERS the timed one.
STEPS  ?= 100
WARMUP ?= 5
ITERS  ?= 10

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64
else
  CFLAGS += -qopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp kinetics.h ../stiff-ode-cuda/mechanism.h ../stiff-ode-cuda/workload.h ../stiff-ode-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --iters $(ITERS) --steps $(STEPS) --warmup $(WARMUP)

.PHONY: clean run
