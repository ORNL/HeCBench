#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = 
LAUNCHER  ?=

KERNEL_DIR := twell_modules

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu \
	 $(KERNEL_DIR)/matmul_d2t.cu \
	 $(KERNEL_DIR)/matmul_t2d.cu \
	 $(KERNEL_DIR)/matmul_gated_t2d.cu \
	 $(KERNEL_DIR)/mlp_twell.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall $(ARCH)

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -ffast-math
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu reference.h $(KERNEL_DIR)/torch_compat.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --reps 100
