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

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

INC = ../attention-paged-cuda

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(INC) -DUSE_ROCM

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

%.o: %.cu $(INC)/attention_dtypes.h \
          $(INC)/attention_kernels.cuh \
          $(INC)/dtype_bfloat16.cuh \
          $(INC)/attention_generic.cuh \
          $(INC)/attention_utils.cuh \
          $(INC)/cuda_compat.h \
          $(INC)/dtype_float32.cuh \
          $(INC)/reference.h \
          kvcache.h \
          Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 8 32 128 4096 128 100
	$(LAUNCHER) ./$(program) 8 32 128 4096 1024 100
	$(LAUNCHER) ./$(program) 8 32 128 4096 8192 100
	$(LAUNCHER) ./$(program) 8 32 128 4096 131072 100
