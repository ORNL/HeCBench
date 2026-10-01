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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../attention-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -munsafe-fp-atomics
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../attention-cuda/reference.h kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 65536 2048 0 1000
	$(LAUNCHER) ./$(program) 65536 2048 1 1000
	$(LAUNCHER) ./$(program) 65536 2048 2 1000
	$(LAUNCHER) ./$(program) 65536 2048 3 1000
	$(LAUNCHER) ./$(program) 8192 8192 0 1000
	$(LAUNCHER) ./$(program) 8192 8192 1 1000
	$(LAUNCHER) ./$(program) 8192 8192 2 1000
	$(LAUNCHER) ./$(program) 8192 8192 3 1000
