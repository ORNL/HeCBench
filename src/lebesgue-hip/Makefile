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
inc = ../lebesgue-cuda

program = main

obj = main.o utils.o kernels.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall 

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

main.o: $(inc)/main.cpp $(inc)/lebesgue.h $(inc)/reference.h
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: $(inc)/utils.cpp $(inc)/lebesgue.h
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) 501 2

run: $(program)
	$(LAUNCHER) ./$(program) 1000000 2
