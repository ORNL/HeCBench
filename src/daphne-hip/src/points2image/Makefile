#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
SHARED_DATA_PATH = ../../../daphne-cuda/data
SHARED_SRC_PATH = ../../../daphne-cuda/src
LAUNCHER        ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = $(SHARED_SRC_PATH)/common/main.cpp kernel.cu

obj = main.o kernel.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          -I$(SHARED_SRC_PATH)/include \
          -I$(SHARED_SRC_PATH)/points2image

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
all: $(program) checkdata

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

kernel.o: kernel.cu $(SHARED_SRC_PATH)/points2image/datatypes.h \
          $(SHARED_SRC_PATH)/include/benchmark.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: $(SHARED_SRC_PATH)/common/main.cpp \
        $(SHARED_SRC_PATH)/include/benchmark.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

checkdata:
ifeq ($(wildcard $(SHARED_DATA_PATH)/p2i_input.dat),)
	$(warning p2i_input.dat not found. Did you forget to extract the test data?)
endif
ifeq ($(wildcard $(SHARED_DATA_PATH)/p2i_output.dat),)
	$(warning p2i_output.dat not found. Did you forget to extract the test data?)
endif

clean:
	rm -rf $(program) $(obj)

# TODO: when p > 1, results mismatch
run: $(program)
	$(LAUNCHER) ./$(program) -p 1
