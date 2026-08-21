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

source = main.cpp kernels.cu utils.cpp

obj = main.o kernels.o utils.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../sundials-cuda

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

kernels.o: kernels.cu ../sundials-cuda/sundials.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: ../sundials-cuda/utils.cpp ../sundials-cuda/sundials.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: ../sundials-cuda/main.cpp ../sundials-cuda/sundials.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 32 20000 6 1000 3
