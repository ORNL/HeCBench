#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
ROCM_PATH ?= /opt/rocm
ARCH      = gfx950
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

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall --offload-arch=$(ARCH) -I$(ROCM_PATH)/include

# Linker Flags
LDFLAGS = -L$(ROCM_PATH)/lib -lhipblaslt

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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

%.o: %.cu helper.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# Default backend is hipblaslt.  The aiter f6gemm kernel uses the same inputs
# and verify():
#   export AITER_ASM_DIR=/path/to/aiter/hsa
#   ./main <repeat> aiter
# AITER_ASM_DIR is the hsa directory of an aiter checkout. The kernel loaded is
# $AITER_ASM_DIR/$ARCH/f6gemm/f6gemm_bf16_per1x32Fp6_dmabig.co.
run: $(program)
	$(LAUNCHER) ./$(program) 100
