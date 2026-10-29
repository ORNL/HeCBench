#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

# Mamba-2 2.7B SSD layer (state-spaces/mamba2-2.7b):
#   d_model=2560, expand=2, headdim=64 -> 80 heads
#   ngroups=1, d_state=128, 2K prefill.
# The benchmark sweeps chunk sizes 32, 64, and 128.
REPEAT ?= 100
BATCH  ?= 8
TOKENS ?= 2048
HEADS  ?= 80
GROUPS ?= 1
HEAD   ?= 64
STATE  ?= 128

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# -fopenmp: the host reference spreads the recurrence over OpenMP threads.
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fopenmp -I../ssm2-cuda

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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

%.o: %.cu ../ssm2-cuda/kernels.h ../ssm2-cuda/reference.h \
          ../ssm2-cuda/ssm2_launch_policy.h ../ssm2-cuda/ssm2_types.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(REPEAT) $(BATCH) $(TOKENS) $(HEADS) $(GROUPS) $(HEAD) $(STATE)

.PHONY: clean run
