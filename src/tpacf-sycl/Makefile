#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

POINTS    = 97178
DATAPARS  = -d ../tpacf-cuda/data/small/Datapnts.1 -p $(POINTS)
RANDPARS  = -r ../tpacf-cuda/data/small/Randompnts -n 100 -q $(POINTS)
ANALPARS  = -b 5 -l 1 -u 10000 -a -j 10
OUTFILE   = ./small_results_test.dat
#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o args.o compute.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -I../tpacf-cuda \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = 

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

args.o: args.cpp args.h
	$(CC) $(CFLAGS) -c $< -o $@

compute.o: compute.cpp model_io.cpp ACF_kernel.cpp histogram_kernel.cpp \
           ../tpacf-cuda/kernel.h args.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp args.h ../tpacf-cuda/kernel.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) $(OUTFILE)

run: $(program)
	$(LAUNCHER) ./$(program) $(DATAPARS) $(RANDPARS) $(ANALPARS) -o $(OUTFILE)
