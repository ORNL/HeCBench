#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_80
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
  CFLAGS += -g -G
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

%.o: %.cu reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 4096 18 256 24 20 20 100
	$(LAUNCHER) ./$(program) 4096 18 256 22 21 21 100
	$(LAUNCHER) ./$(program) 16384 18 256 24 20 20 100
	$(LAUNCHER) ./$(program) 16384 18 256 22 21 21 100

test: $(program)
	$(LAUNCHER) ./$(program) 1000 3 64 8 4 4 100
	$(LAUNCHER) ./$(program) 999 5 128 16 8 8 100
