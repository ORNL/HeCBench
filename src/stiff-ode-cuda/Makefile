#===============================================================================
# User Options
#===============================================================================

CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_80
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

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -Xcompiler -fopenmp -arch=$(ARCH)

LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu kinetics.cuh mechanism.h workload.h reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --iters $(ITERS) --steps $(STEPS) --warmup $(WARMUP)

.PHONY: clean run
