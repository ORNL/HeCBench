ROCM_PATH ?= /opt/rocm
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

source = main.cpp kernels.cu utils.cpp

obj = main.o kernels.o utils.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../spmv-cuda \
          -I$(ROCM_PATH)/include/hipsparse

# Linker Flags
LDFLAGS = -lhipsparse

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

kernels.o: kernels.cu ../spmv-cuda/mv.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: ../spmv-cuda/utils.cpp ../spmv-cuda/mv.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: ../spmv-cuda/main.cpp ../spmv-cuda/mv.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 16777216 10240 100
