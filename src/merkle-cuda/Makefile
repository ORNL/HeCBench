#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = no
ARCH      = sm_60
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu bench_merkle_tree.cu merkle_tree.cu \
         rescue_prime.cu ff_p.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH) --expt-relaxed-constexpr

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# Verification Flags
ifeq ($(VERIFY),yes)
  CFLAGS  += -DDEBUG
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -dc $< -o $@

main.cu: bench_merkle_tree.hpp
bench_merkle_tree.cu: bench_merkle_tree.hpp
merkle_tree.cu: merkle_tree.hpp
rescue_prime.cu: rescue_prime.hpp
ff_p.cu: ff_p.hpp

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
