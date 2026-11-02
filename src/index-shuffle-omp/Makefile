#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

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

ifneq ($(DEVICE),gpu)
  $(error index-shuffle-omp requires DEVICE=gpu)
endif
CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../index-shuffle-cuda/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Tokens in {128, 2048, 4096, 8192}, experts in {16, 128}, top-k in {1, 2, 4}.
TOKENS  = 128 2048 4096 8192
EXPERTS = 16 128
TOPKS   = 1 2 4

run: $(program)
	@set -e; \
	for t in $(TOKENS); do for e in $(EXPERTS); do for k in $(TOPKS); do \
	  $(LAUNCHER) ./$(program) $$t $$e $$k 1000; \
	done; done; done
