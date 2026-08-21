#===============================================================================
# User Options
#===============================================================================

MPI_ROOT     ?= /usr/lib/x86_64-linux-gnu/openmpi
ROCM_PATH    ?= /opt/rocm
HIPBLAS_ROOT ?= $(ROCM_PATH)
HIPRT_ROOT   ?= $(ROCM_PATH)

# Compiler can be set below, or via environment variable
CC        = hipcc
LD        = mpiCC
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER ?= mpirun -n 4

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
CFLAGS := $(EXTRA_CFLAGS) -I$(MPI_ROOT)/include -std=c++17 -Wall -fPIE \
          -I$(ROCM_PATH)/include/hipblas

# Linker Flags
LDFLAGS = -L$(HIPBLAS_ROOT)/lib -lhipblas -L$(HIPRT_ROOT)/lib -lamdhip64 

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
	$(LD) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --num_vector 4000 --num_field 90000 --num_iterations 10
