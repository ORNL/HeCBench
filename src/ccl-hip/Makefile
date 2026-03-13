#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
MPI_ROOT  = $(HOME)/ompi_for_gpu/ompi
LAUNCHER  = $(HOME)/ompi_for_gpu/ompi/bin/mpirun --mca pml ucx --mca osc ucx \
            --mca coll_ucc_enable 1 --mca coll_ucc_priority 100 -n 2

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -I$(MPI_ROOT)/include -Wall

# Linker Flags
LDFLAGS = -L$(MPI_ROOT)/lib -lmpi -lrccl -L/opt/rocm/lib/ -lamdhip64

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

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 50
