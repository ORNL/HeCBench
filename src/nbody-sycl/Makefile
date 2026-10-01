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

source = main.cpp GSimulation.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../nbody-cuda \
          -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

# The host reference in ../nbody-cuda/GSimulationReference.hpp is the golden
# result the device output is checked against, and it is compiled only into
# main.o. icpx defaults to -fp-model=fast, which allows FMA contraction and
# reassociation in that host accumulation and makes the single-precision
# reference drift past the 1e-3 error bound (see
# https://github.com/intel/llvm/issues/21197). Force precise, non-contracted FP
# for the reference so it matches the gcc/nvcc host result. These flags are
# applied ONLY to main.o so the benchmarked device kernel in GSimulation.o keeps
# its default (fast-math) code generation.
#
# The FP-model spelling differs by compiler:
#   - icpx accepts the Intel driver flag -fp-model (single f).
#   - intel/llvm open-source clang++ only accepts the Clang flag -ffp-model
#     (double f); -fp-model is rejected as an unknown argument.
#   - AdaptiveCpp (acpp) is not guaranteed to accept -ffp-model, so use the
#     portable -ffp-contract=off there (it does not default to fast-math).
ifneq (,$(findstring icpx,$(CC)))
  FPFLAGS = -fp-model=precise -ffp-contract=off
else ifeq ($(VENDOR),AdaptiveCpp)
  FPFLAGS = -ffp-contract=off
else
  FPFLAGS = -ffp-model=precise -ffp-contract=off
endif

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

# main.o holds the host reference: compile it with precise, non-contracted FP.
main.o: main.cpp GSimulation.hpp Particle.hpp type.hpp \
           ../nbody-cuda/GSimulationReference.hpp
	$(CC) $(CFLAGS) $(FPFLAGS) -c $< -o $@

# GSimulation.o holds the device kernel: keep default code generation.
%.o: %.cpp GSimulation.hpp Particle.hpp type.hpp \
           ../nbody-cuda/GSimulationReference.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) 16000 10
