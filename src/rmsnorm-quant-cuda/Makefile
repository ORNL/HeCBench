#===============================================================================
# User Options
#===============================================================================

CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_90
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

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH) \
          -I../rmsnorm-cuda

LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu common.h fp8_e4m3.h reference.h ../rmsnorm-cuda/utils.cuh \
     ../rmsnorm-cuda/reduce.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1 12288 1000
	$(LAUNCHER) ./$(program) 1024 12288 1000
