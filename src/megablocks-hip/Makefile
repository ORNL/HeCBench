#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
ROCM_PATH ?= /opt/rocm
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(ROCM_PATH)/include

# Linker Flags
LDFLAGS = -L$(ROCM_PATH)/lib -lhipblas -lhipblaslt

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

%.o: %.cu ../megablocks-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Usage: ./main [repeat] [num_experts] [hidden_in] [hidden_out] [avg_tokens]
run: $(program)
	$(LAUNCHER) ./$(program) 100 8 4096 14336 32
	$(LAUNCHER) ./$(program) 100 64 2048 1408 16
	$(LAUNCHER) ./$(program) 100 128 5120 1536 8
	# low tokens/expert => empty experts exercise the zero-out + skip path
	$(LAUNCHER) ./$(program) 100 64 2048 1408 4
	$(LAUNCHER) ./$(program) 100 128 5120 1536 2
