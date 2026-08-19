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
ONEMKL    = no
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl -fno-sycl-id-queries-fit-in-int \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -I../bgmv-cuda

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

ifeq ($(CUDA), yes)
  ONEMKL = yes
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  ONEMKL = yes
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

%.o: %.cpp ../bgmv-cuda/reference.h ../bgmv-cuda/config.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --op bgmv_shrink --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
