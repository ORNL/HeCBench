#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu hisparse_kernels.cu

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

main.o: main.cu hisparse_kernels.h reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

hisparse_kernels.o: hisparse_kernels.cu hisparse_kernels.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean run

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 8 256 100
	$(LAUNCHER) ./$(program) 32 1024 100
	$(LAUNCHER) ./$(program) 1 2048 100
	$(LAUNCHER) ./$(program) 64 2048 100
	$(LAUNCHER) ./$(program) 128 2048 100
	$(LAUNCHER) ./$(program) 256 2048 100
