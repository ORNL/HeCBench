#===============================================================================
# User Options
#===============================================================================

CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

NX     ?= 128
NY     ?= 128
NZ     ?= 128
NODES  ?= 512
WARMUP ?= 1000
ITERS  ?= 1000

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../condor-cuda

LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS += -fopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../condor-cuda/condor_reference.hpp ../condor-cuda/condor_host.hpp \
           Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --nx $(NX) --ny $(NY) --nz $(NZ) \
        --nodes $(NODES) --iters $(ITERS) --warmup $(WARMUP)

.PHONY: clean run
