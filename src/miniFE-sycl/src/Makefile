#-----------------------------------------------------------------------
# This file compiles for OpenMP and MPI hybrid operations using the GNU
# compile chain.

MINIFE_TYPES =  \
        -DMINIFE_SCALAR=double   \
        -DMINIFE_LOCAL_ORDINAL=int      \
        -DMINIFE_GLOBAL_ORDINAL=int \
	-DMINIFE_RESTRICT=__restrict__

MINIFE_MATRIX_TYPE = -DMINIFE_CSR_MATRIX

#-----------------------------------------------------------------------
CXX       = clang++
CC        = clang
OPTIMIZE  = yes
DEBUG     = no
GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
# Use sycl_ext_oneapi_enqueue_functions (event-less kernel submission) to
# reduce per-launch overhead of the many small CG kernels.
ENQUEUE_FUNCTIONS = yes
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

CPPFLAGS = -I. -I../utils -I../fem $(MINIFE_TYPES) $(MINIFE_MATRIX_TYPE) \
           -std=c++17 -Wall -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CPPFLAGS_TMP := $(CPPFLAGS)
    CPPFLAGS = $(filter-out -fsycl, $(CPPFLAGS_TMP))
endif

ifeq ($(CUDA), yes)
  # Auto-detect the installed NVIDIA GPU's compute capability (e.g. 12.0 -> sm_120)
  # so we build natively for the target device. Overridable on the command line
  # via `make CUDA_ARCH=sm_XX` (command-line assignments take precedence).
  DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
  ifneq ($(DETECTED_CC),)
    CUDA_ARCH := sm_$(DETECTED_CC)
  endif
  $(info miniFE: building for CUDA arch $(CUDA_ARCH))
  CPPFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
              -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CPPFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	      -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  CPPFLAGS += -DMINIFE_DEBUG
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

CXXFLAGS = $(CFLAGS)

ifeq ($(GPU),yes)
  CPPFLAGS +=-DUSE_GPU
endif

ifeq ($(ENQUEUE_FUNCTIONS),yes)
  # The enqueue_functions extension needs the new offloading driver so that its
  # free-function kernels are registered in these separate compile + link builds.
  CPPFLAGS += -DUSE_ENQUEUE_FUNCTIONS --offload-new-driver
endif

LIBS=

# The MPICH_IGNORE_CXX_SEEK macro is required for some mpich versions,
# such as the one on my cygwin machine.

include make_targets
