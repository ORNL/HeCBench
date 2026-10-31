#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_70
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

%.o: %.cu kernels.h reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Tokens in {4096, 32768}, experts in {16, 128}, top-k in {1, 2, 4}.
# Optional 5th argument: fused (default) or split.
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
