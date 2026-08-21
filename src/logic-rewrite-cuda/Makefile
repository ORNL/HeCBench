#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 --use_fast_math 
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

VERIFY_SRC = verify.cpp
verify: $(VERIFY_SRC)
	$(CXX) -O3 -std=c++17 $(VERIFY_SRC) -o verify

run: $(program) verify
	$(LAUNCHER) ./$(program) -c "read ./benchmarks/arithmetic/hyp.aig; resyn2; write hyp_resyn2.aig"
	./verify ./benchmarks/arithmetic/hyp.aig hyp_resyn2.aig
