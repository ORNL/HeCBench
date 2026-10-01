#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
BOOST     = $(HOME)/boost_1_76_0
ARCH      = sm_60
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu kernels.cu mttkrp_cpu.cu 


obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -I$(BOOST) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = -lpthread

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

%.o: %.cu mttkrp_cpu.h mttkrp_gpu.h util.h   
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

# generate a 4D tensor with random values
output.tns: generate_tensor.py
	python generate_tensor.py output.tns 2%4096 2%4096 2%4096 2%512

test: toy.tns $(program)
	$(LAUNCHER) ./$(program) -i toy.tns -m 0 -R 32 -f 128 -w 1
  
run: output.tns $(program)
	$(LAUNCHER) ./$(program) -i output.tns -m 0 -R 32 -f 128 -w 4
