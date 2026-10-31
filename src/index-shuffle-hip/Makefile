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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker Flags
LDFLAGS = 

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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../index-shuffle-cuda/main.cu ../index-shuffle-cuda/kernels.h \
     ../index-shuffle-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Tokens in {4096, 32768}, experts in {16, 128}, top-k in {1, 2, 4}.
# Optional 5th argument: fused (default) or split. The driver is the CUDA one.
TOKENS  = 4096 32768
EXPERTS = 16 128
TOPKS   = 1 2 4

run: $(program)
	@set -e; \
	for t in $(TOKENS); do for e in $(EXPERTS); do for k in $(TOPKS); do \
	  $(LAUNCHER) ./$(program) $$t $$e $$k 1000; \
	done; done; done

run-split: $(program)
	@set -e; \
	for t in $(TOKENS); do for e in $(EXPERTS); do for k in $(TOPKS); do \
	  $(LAUNCHER) ./$(program) $$t $$e $$k 1000 split; \
	done; done; done
