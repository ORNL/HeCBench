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
SPATH = ../boxfilter-sycl

program = main

obj = main.o shrUtils.o MedianFilterHost.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH) \
          -I$(SPATH)

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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cu MedianFilter.cu $(SPATH)/shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

MedianFilterHost.o : MedianFilterHost.cpp
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o : $(SPATH)/shrUtils.cpp $(SPATH)/shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../medianfilter-sycl/data/SierrasRGB.ppm 1000
