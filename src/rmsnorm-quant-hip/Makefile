#===============================================================================
# User Options
#===============================================================================

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

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          -I../rmsnorm-quant-cuda -I../rmsnorm-hip

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

%.o: %.cu common.h ../rmsnorm-quant-cuda/fp8_e4m3.h \
     ../rmsnorm-quant-cuda/reference.h ../rmsnorm-hip/utils.cuh \
     ../rmsnorm-hip/reduce.cuh Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1 12288 1000
	$(LAUNCHER) ./$(program) 1024 12288 1000
