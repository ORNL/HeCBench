#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = KeccakF.cpp KeccakTreeCPU.cpp KeccakTreeGPU.cpp Test.cpp main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall 

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): KeccakTreeCPU.o KeccakTreeGPU.o Test.o main.o KeccakF.o Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

KeccakTreeCPU.o: KeccakTreeCPU.cpp KeccakTreeCPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

KeccakTreeGPU.o: KeccakTreeGPU.cpp KeccakTreeGPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

Test.o: Test.cpp KeccakTreeCPU.h KeccakTreeGPU.h KeccakF.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp KeccakTreeCPU.h KeccakTreeGPU.h Test.h KeccakTypes.h KeccakTree.h
	$(CC) $(CFLAGS) -c $< -o $@

KeccakF.o: KeccakF.cpp KeccakTree.h KeccakF.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program)
