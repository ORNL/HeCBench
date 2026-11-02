#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_80
LAUNCHER  ?=

# Representative workload from NVlabs/kda issue 6.
# `python vdn_delta_inverse_ref.py` with no arguments is exactly this shape:
#   101 frames x 7 heads = 707 matrices, d = 128, 1008 tokens per frame.
# The model runs that call 50 times per forward. Shorter clips lower N
# toward ~100, for example: make run FRAMES=15 HEADS=7
# Frames and heads change only the number of matrices. They are not
# compile-time kernel parameters.
FRAMES ?= 101
HEADS  ?= 7
TOKENS ?= 1008
DIM    ?= 128
REPEAT ?= 100

# make run launches the same draw at beta-scale 1, 50, and 100. The seed,
# frames, heads, and tokens stay the same, so K, V, and alpha are the same.
# beta-scale multiplies every per-token beta, and A and B are linear in beta.
# Alpha does not change. Larger A makes the fp32 dots in J = B (I+A)^{-1}
# accumulate bigger partial products. Each process must stay finite and
# within 1e-6 of the fp64 inverse.

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu fused.cu libraries.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags. fp32 Cholesky and the inverse products must not use TF32
# or fast-math sqrt/div approximations.
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = -lcublas -lcusolver

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G -DDEBUG
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

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: host_common.hpp reference.h fused.h libraries.h
fused.o: fused.h local_layout.hpp
libraries.o: libraries.h library_batch.cuh

clean:
	rm -rf $(program) $(obj)

.PHONY: clean run

run: $(program)
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 1
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 50
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 100
