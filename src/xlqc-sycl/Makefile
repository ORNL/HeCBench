#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
GSL_INC   ?= $(shell gsl-config --cflags)
GSL_LIB   ?= $(shell gsl-config --libs)
LAUNCHER  =

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

ifeq ($(GSL_LIB),)
$(error gsl-config not found; install libgsl-dev or set GSL_INC/GSL_LIB)
endif

#===============================================================================
# Program name & source code list
#===============================================================================

program = main


obj=basis.o scf.o main.o crys.o cints.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -I../xlqc-cuda -I../xlqc-cuda/int_lib $(GSL_INC) \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -std=c++17 -Wall -fsycl

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = $(GSL_LIB) 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp cuda_rys_sp.cpp cuda_rys_dp.cpp rys.h
	$(CC) -c -o $@ $< $(CFLAGS)

scf.o: ../xlqc-cuda/scf.cc
	$(CC) -c -o $@ $< $(CFLAGS)

basis.o: ../xlqc-cuda/basis.cc
	$(CC) -c -o $@ $< $(CFLAGS)

crys.o: ../xlqc-cuda/int_lib/crys.cc
	$(CC) -c -o $@ $< $(CFLAGS)

cints.o: ../xlqc-cuda/int_lib/cints.cc
	$(CC) -c -o $@ $< $(CFLAGS)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) sp
	$(LAUNCHER) ./$(program) dp
