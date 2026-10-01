#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
BOOST     = $(HOME)/boost_1_76_0
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = ../mmcsf-cuda/main.cu kernels.cu ../mmcsf-cuda/mttkrp_cpu.cu


obj = main.o kernels.o mttkrp_cpu.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -I$(BOOST) -std=c++17 -I../mmcsf-cuda

# Linker Flags
LDFLAGS = -lpthread

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

main.o: ../mmcsf-cuda/main.cu ../mmcsf-cuda/mttkrp_cpu.h mttkrp_gpu.h ../mmcsf-cuda/util.h
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cu ../mmcsf-cuda/mttkrp_cpu.h mttkrp_gpu.h ../mmcsf-cuda/util.h
	$(CC) $(CFLAGS) -c $< -o $@

mttkrp_cpu.o : ../mmcsf-cuda/mttkrp_cpu.cu ../mmcsf-cuda/mttkrp_cpu.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

output.tns: ../mmcsf-cuda/generate_tensor.py
	python ../mmcsf-cuda/generate_tensor.py output.tns 2%4096 2%4096 2%4096 2%512

test: ../mmcsf-cuda/toy.tns $(program)
	$(LAUNCHER) ./$(program) -i ../mmcsf-cuda/toy.tns -m 0 -R 32 -f 128 -w 1

run: output.tns $(program)
	$(LAUNCHER) ./$(program) -i output.tns -m 0 -R 32 -f 128 -w 4
