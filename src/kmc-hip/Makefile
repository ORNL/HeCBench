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

obj = main.o svm.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -Wno-unused-result -I../kmc-cuda \
          -I$(ROCM_PATH)/include/hipblas

# Linker Flags
LDFLAGS = -L$(ROCM_PATH)/hipblas/lib/ -lhipblas 

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

main.o: main.cu ../kmc-cuda/driver.c ../kmc-cuda/svm.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

svm.o: ../kmc-cuda/svm.cpp ../kmc-cuda/driver.c ../kmc-cuda/svm.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -v 10 gisette_scale
