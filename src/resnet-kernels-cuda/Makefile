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

source = main.cu util.cu \
         Kernel256_winograd.cu \
         Kernel128_winograd.cu \
         Kernel256_one.cu \
         Kernel128_one.cu

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
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) data

data/input_14_1_128.bin:
	mkdir -p data
	python3 data_generator.py

run: $(program) data/input_14_1_128.bin
	$(LAUNCHER) ./$(program) 0 100
	$(LAUNCHER) ./$(program) 1 100
	$(LAUNCHER) ./$(program) 2 100
	$(LAUNCHER) ./$(program) 3 100
	$(LAUNCHER) ./$(program) 4 100
	$(LAUNCHER) ./$(program) 5 100
