#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?=
LAUNCHER  ?=
FIXTURE   ?= ../data/hydragnn-sc26/mace_pipeline_capture.macepipe

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

ifneq ($(ARCH),)
  CFLAGS += --offload-arch=$(ARCH)
endif

# Linker Flags
LDFLAGS = -lrocblas

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

%.o: %.cu mace_pipeline.hpp mace_nlist.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# The timed workload is a synthetic batch at the published SC26 dimensions:
# 128 graphs, 2726 atoms, 45974 edges. Correctness is validated on the captured
# fixture first.
run: $(program)
	$(LAUNCHER) ./$(program) --fixture $(FIXTURE) --warmup 100 --repeat 100
