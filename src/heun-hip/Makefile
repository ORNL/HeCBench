#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

# Production sampler shape: one forecast, 99 HRRR variables, 512 x 640.
BATCH     ?= 1
CHANNELS  ?= 99
HEIGHT    ?= 512
WIDTH     ?= 640
STEPS     ?= 18

program = main

source = main.cu

obj = $(source:.cu=.o)

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker Flags
LDFLAGS = -lMIOpen -pthread

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

all: $(program)

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../heun-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: all
	$(LAUNCHER) ./$(program) 100 $(BATCH) $(CHANNELS) $(HEIGHT) $(WIDTH) $(STEPS)
