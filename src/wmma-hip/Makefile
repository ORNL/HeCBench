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

%.o: %.cu reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 0 16 16 64 100 1
	$(LAUNCHER) ./$(program) 0 32 32 64 100 1
	$(LAUNCHER) ./$(program) 0 256 256 256 100 1
	$(LAUNCHER) ./$(program) 1 256 256 256 100 1
	$(LAUNCHER) ./$(program) 0 512 512 512 100 1
	$(LAUNCHER) ./$(program) 1 512 512 512 100 1
	$(LAUNCHER) ./$(program) 0 1024 1024 1024 100 0
	$(LAUNCHER) ./$(program) 1 1024 1024 1024 100 0
	$(LAUNCHER) ./$(program) 0 4096 4096 4096 100 0
	$(LAUNCHER) ./$(program) 1 4096 4096 4096 100 0
	$(LAUNCHER) ./$(program) 0 8192 8192 8192 100 0
	$(LAUNCHER) ./$(program) 1 8192 8192 8192 100 0
