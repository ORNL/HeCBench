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

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

inc = ../cmp-cuda

obj = main.o reference.o su_cdp.o su_gather.o su_trace.o log.o parser.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(inc) -fsycl \
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

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

log.o : $(inc)/log.cpp $(inc)/log.hpp
	$(CC) $(CFLAGS) -c $< -o $@

parser.o: $(inc)/parser.cpp $(inc)/parser.hpp  
	$(CC) $(CFLAGS) -c $< -o $@

su_cdp.o : $(inc)/su_cdp.cpp $(inc)/su_cdp.hpp $(inc)/log.hpp
	$(CC) $(CFLAGS) -c $< -o $@

su_gather.o : $(inc)/su_gather.cpp $(inc)/su_gather.hpp $(inc)/log.hpp $(inc)/su_trace.hpp
	$(CC) $(CFLAGS) -c $< -o $@

su_trace.o : $(inc)/su_trace.cpp $(inc)/su_trace.hpp $(inc)/log.hpp
	$(CC) $(CFLAGS) -c $< -o $@

main.o : main.cpp $(inc)/log.hpp $(inc)/utils.hpp $(inc)/parser.hpp $(inc)/su_gather.hpp
	$(CC) $(CFLAGS) -c $< -o $@

reference.o : $(inc)/reference.cpp $(inc)/utils.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -c0 1.98e-7 -c1 1.77e-6 -nc 5 -aph 600 -tau 0.002 -v 2 \
                     -i $(inc)/data/simple-synthetic.su
