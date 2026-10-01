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

source = main.cpp device.cu

obj = main.o device.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../pcc-cuda -I $(ROCM_PATH)/include/hipblas

# Linker Flags
LDFLAGS = -lhipblas

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

main.o: ../pcc-cuda/main.cpp ../pcc-cuda/device.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

device.o: device.cu ../pcc-cuda/device.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) corrs.txt

run: $(program)
	$(LAUNCHER) ./$(program) 90112 165
