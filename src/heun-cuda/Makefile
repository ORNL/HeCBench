#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?=

# Production sampler shape: one forecast, 99 HRRR variables, 512 x 640.
BATCH     ?= 1
CHANNELS  ?= 99
HEIGHT    ?= 512
WIDTH     ?= 640
STEPS     ?= 18

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# cuDNN is outside nvcc's default search path when it comes from the
# pip/conda nvidia-cudnn package. Set CUDNN_ROOT to the prefix that
# contains include/cudnn.h and lib/libcudnn.so* to override detection.
CUDNN_ROOT ?= $(shell python3 -c "import nvidia.cudnn as c; print(next(iter(c.__path__)))" 2>/dev/null)

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = -lpthread
ifneq ($(wildcard $(CUDNN_ROOT)/include/cudnn.h),)
  CFLAGS += -I$(CUDNN_ROOT)/include
  ifneq ($(wildcard $(CUDNN_ROOT)/lib/libcudnn.so),)
    CUDNN_LIBFILE := $(CUDNN_ROOT)/lib/libcudnn.so
  else
    CUDNN_LIBFILE := $(firstword $(wildcard $(CUDNN_ROOT)/lib/libcudnn.so.*))
  endif
  LDFLAGS += -Xlinker $(CUDNN_LIBFILE) -Xlinker -rpath -Xlinker $(CUDNN_ROOT)/lib
else
  LDFLAGS += -lcudnn
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

#===============================================================================
# Targets to Build
#===============================================================================

all: $(program)

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: all
	$(LAUNCHER) ./$(program) 100 $(BATCH) $(CHANNELS) $(HEIGHT) $(WIDTH) $(STEPS)
