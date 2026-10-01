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
OPENMP_INCLUDE ?= /path/to/openmp/include
OPENMP_LIB ?= /path/to/openmp/lib

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = src/main.cpp src/seed.cpp src/score.cpp src/logan_functions.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -Isrc -I$(OPENMP_INCLUDE) \
          -fopenmp -fsycl

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = -L$(OPENMP_LIB) -lomp

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
  CFLAGS += -O3 -ffast-math
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

src/main.o: src/main.cpp src/logan.hpp
	$(CC) $(CFLAGS) -c $< -o $@

src/seed.o: src/seed.cpp src/seed.hpp
	$(CC) $(CFLAGS) -c $< -o $@

src/score.o: src/score.cpp src/score.hpp
	$(CC) $(CFLAGS) -c $< -o $@

src/logan_functions.o: src/logan_functions.cpp \
                       src/logan_functions.hpp src/seed.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

demo: $(program)
	./$(program) ../logan-cuda/inputs/example.txt 17 21 1

run: $(program)
	$(LAUNCHER) ./$(program) ../logan-cuda/inputs/100k.txt 17 100 1
