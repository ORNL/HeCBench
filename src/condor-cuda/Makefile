#===============================================================================
# User Options
#===============================================================================

CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_80
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

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall \
          -Xcompiler -fopenmp -arch=$(ARCH)

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

%.o: %.cu condor_reference.hpp condor_host.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --nx $(NX) --ny $(NY) --nz $(NZ) \
        --nodes $(NODES) --iters $(ITERS) --warmup $(WARMUP)

.PHONY: clean run
