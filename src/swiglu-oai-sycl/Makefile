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
CUDA_ARCH ?= sm_90
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

# Run options. The default run mirrors the representative CUDA/HIP paths.
ROWS   ?= 8192
DIM    ?= 1024
REPEAT ?= 100

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../swiglu-oai-cuda -fsycl

ifneq ($(strip $(GCC_TOOLCHAIN)),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
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
  # The mxfp4 kernel is also instantiated with a required sub-group size of
  # 32, which amdgcn ignores. main() only selects that instantiation on
  # devices that report the size as supported.
  CFLAGS += -Wno-incorrect-sub-group-size
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

$(program): $(source) ../swiglu-oai-cuda/reference.h Makefile
	$(CC) $(CFLAGS) $(source) -o $@ $(LDFLAGS)

clean:
	rm -rf $(program) $(obj)

.PHONY: clean run

run: $(program)
	@run_case() { \
	  output="$$($(LAUNCHER) ./$(program) "$$@" 2>&1)"; status=$$?; \
	  printf '%s\n' "$$output"; \
	  if [ $$status -ne 0 ] || \
	     ! printf '%s\n' "$$output" | \
	       awk '$$0 == "PASS" || /^Skipped: / { ok=1 } END { exit !ok }'; then \
	    return 1; \
	  fi; \
	}; \
	set -e; \
	run_case $(ROWS) $(DIM) $(REPEAT) bf16 none none; \
	run_case $(ROWS) $(DIM) $(REPEAT) fp16 none fp16; \
	run_case $(ROWS) $(DIM) $(REPEAT) bf16 fp8 none; \
	run_case $(ROWS) $(DIM) $(REPEAT) bf16 mxfp4 none; \
	run_case $(ROWS) $(DIM) $(REPEAT) mxfp8 fp8 bf16
