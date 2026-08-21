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

source = main.cpp kernel_gpu.cu kernel_cpu.cpp

obj = main.o kernel_gpu.o kernel_cpu.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := -std=c++17 -Wall -I../pso-cuda

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

kernel_gpu.o: kernel_gpu.cu ../pso-cuda/kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

kernel_cpu.o: ../pso-cuda/kernel_cpu.cpp ../pso-cuda/kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: ../pso-cuda/main.cpp ../pso-cuda/kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 30 10000
