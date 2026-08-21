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

source = main.cu shmem_kernels.cu

obj = main.o shmem_kernels.o

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

$(program): main.o shmem_kernels.o
	$(CC) $(CFLAGS) $^ -o $@ ${LFLAGS}

main.o: main.cu shmem_kernels.h
	$(CC) -c $(CFLAGS) $<

shmem_kernels.o: shmem_kernels.cu
	$(CC) -c $(CFLAGS) $<

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 1000


