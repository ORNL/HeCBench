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

source = aig_manager.cu balance.cu print.cu refactor_core.cu \
         refactor.cu refactor_mffc.cu rewrite.cu \
         strash.cu truth.cu command_manager.cu main.cu


obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -DHIP_ENABLE_WARP_SYNC_BUILTINS=1

# Linker Flags
LDFLAGS = -lpthread

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu *.h *.cuh
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) verify *.aig

VERIFY_SRC = ../logic-rewrite-cuda/verify.cpp
verify: $(VERIFY_SRC)
	$(CXX) -O3 -std=c++17 $(VERIFY_SRC) -o verify

run: $(program) verify
	$(LAUNCHER) ./$(program) -c "read ../logic-rewrite-cuda/benchmarks/arithmetic/hyp.aig; resyn2; write hyp_resyn2.aig"
	./verify ../logic-rewrite-cuda/benchmarks/arithmetic/hyp.aig hyp_resyn2.aig
