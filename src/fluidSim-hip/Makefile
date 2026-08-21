#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = yes
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================
program = main

obj = main.o kernels.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../fluidSim-cuda

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

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

kernels.o: kernels.cu ../fluidSim-cuda/utils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: ../fluidSim-cuda/main.cu ../fluidSim-cuda/reference.h \
        ../fluidSim-cuda/utils.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 10000
