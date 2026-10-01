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

source = main.cu kernel.cu host_task.cpp

obj = main.o kernel.o host_task.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../pad-cuda

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

%.o: %.cu ../pad-cuda/kernel.h \
          ../pad-cuda/support/common.h \
          ../pad-cuda/support/partitioner.h \
          ../pad-cuda/support/setup.h \
          ../pad-cuda/support/verify.h \
          Makefile
	$(CC) $(CFLAGS) -c $< -o $@

host_task.o: ../pad-cuda/host_task.cpp \
             ../pad-cuda/kernel.h \
             ../pad-cuda/support/common.h \
             ../pad-cuda/support/partitioner.h \
             Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -a 0.1
