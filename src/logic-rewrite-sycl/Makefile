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

source = aig_manager.cpp balance.cpp print.cpp refactor_core.cpp \
         refactor.cpp refactor_mffc.cpp rewrite.cpp \
         strash.cpp truth.cpp command_manager.cpp main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I. -fsycl \
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
  # amdgcn libspirv lacks the __spirv_GroupNonUniform*Shuffle* builtins that
  # oneDPL work-group scan relies on
  source += amd_spirv_shim.cpp
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
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp *.h *.hpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) verify *.aig

VERIFY_SRC = ../logic-rewrite-cuda/verify.cpp
verify: $(VERIFY_SRC)
	$(CXX) -O3 -std=c++17 $(VERIFY_SRC) -o verify

run: $(program) verify
	$(LAUNCHER) ./$(program) -c "read ../logic-rewrite-cuda/benchmarks/arithmetic/hyp.aig; resyn2; write hyp_resyn2.aig"
	./verify ../logic-rewrite-cuda/benchmarks/arithmetic/hyp.aig hyp_resyn2.aig
