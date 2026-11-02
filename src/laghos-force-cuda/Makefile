#===============================================================================
# User Options
#===============================================================================

CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= sm_90
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

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -arch=$(ARCH)

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

%.o: %.cu reference.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Laghos specialized 3D force sizes: Q1-Q0, Q2-Q1, Q3-Q2, Q4-Q3.
run: $(program)
	$(LAUNCHER) ./$(program) --points 8
	$(LAUNCHER) ./$(program) --points 64
	$(LAUNCHER) ./$(program) --points 216
	$(LAUNCHER) ./$(program) --points 512

.PHONY: clean run
