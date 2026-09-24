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

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DREFDATA_TWOJ=14 -I. -I../testSNAP-omp \
          --gcc-toolchain=$(GCC_TOOLCHAIN) \
          -std=c++17 -Wall -fsycl

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
# SNAP reference headers are stored in testSNAP-omp (DVC: refdata.tar.gz)
REFDATA_DIR := ../testSNAP-omp
REFDATA_TGZ := $(REFDATA_DIR)/refdata.tar.gz
REFDATA_HDR := $(REFDATA_DIR)/refdata_2J2_W.h

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../testSNAP-omp/snap.h utils.cpp $(REFDATA_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(REFDATA_HDR): $(wildcard $(REFDATA_TGZ))
	@test -f $(REFDATA_TGZ) || { \
	  echo "Missing $(REFDATA_TGZ). From the repository root run: dvc pull src/testSNAP-omp/refdata.tar.gz" >&2; \
	  exit 1; }
	tar -xzf $(REFDATA_TGZ) -C $(REFDATA_DIR)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --nsteps 100
