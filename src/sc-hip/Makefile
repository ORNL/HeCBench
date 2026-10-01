#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
DP        = no
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu device_sc.cu host_sc.cpp

obj = main.o device_sc.o host_sc.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../sc-cuda

# Linker Flags
LDFLAGS = -lpthread

ifeq ($(DP),yes)
  CFLAGS += -DDYNAMIC_PARTITION
endif

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

%.o: %.cu kernel.h \
          ../sc-cuda/support/common.h \
          ../sc-cuda/support/partitioner.h \
          ../sc-cuda/support/setup.h \
          ../sc-cuda/support/verify.h \
          Makefile
	$(CC) $(CFLAGS) -c $< -o $@

host_sc.o: ../sc-cuda/host_sc.cpp kernel.h \
           ../sc-cuda/support/common.h \
           ../sc-cuda/support/partitioner.h \
           Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -a 0.1
