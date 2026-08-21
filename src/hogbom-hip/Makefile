#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
VERIFY    = no
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

src = ../hogbom-cuda

obj = main.o kernels.o reference.o timer.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../include -I$(src)


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

ifeq ($(VERIFY),yes)
  CFLAGS += -DVERIFY
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: $(src)/main.cpp $(src)/utils.h $(src)/timer.h $(src)/reference.h $(src)/kernels.h
	$(CC) $(CFLAGS) -c $< -o $@

reference.o: $(src)/reference.cpp $(src)/reference.h 
	$(CC) $(CFLAGS) -c $< -o $@

timer.o: $(src)/timer.cpp $(src)/timer.h 
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cu $(src)/kernels.h $(src)/timer.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(src)/data/dirty_4096.img $(src)/data/psf_4096.img 1000
