#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
# __nv_fp8_e4m3 (cuda_fp8.h) needs CUDA >= 11.8; the native FP4 conversions
# (cuda_fp4.h) need CUDA >= 12.8, and older toolkits fall back to the software
# E2M1/E8M0 codecs in main.cu. FP4 conversion is native on
# architecture-specific Blackwell targets such as sm_100a; other targets use
# CUDA's emulation path. sm_90 keeps this benchmark runnable on Hopper while
# exercising native FP8 support.
ARCH      = sm_90
LAUNCHER  ?=

# Run options. The default run covers five representative production paths.
# Override the shape or repeat count on the make command line, for example:
#   make run ROWS=128 DIM=256 REPEAT=10
ROWS   ?= 8192
DIM    ?= 1024
REPEAT ?= 100

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS =

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

.PHONY: clean run

#
# Standard LLM inference: BF16 activation, no bias, no output quantization.
# Bias-bearing FP16 model.
# vLLM/SGLang-style fused SwiGLU followed by FP8 activation quantization.
# Blackwell-style fused SwiGLU followed by MXFP4 activation quantization.
# Low-precision MoE path with MXFP8 input, FP8 output, and BF16 bias.
#
run: $(program)
	@run_case() { \
	  output="$$($(LAUNCHER) ./$(program) "$$@" 2>&1)"; status=$$?; \
	  printf '%s\n' "$$output"; \
	  if [ $$status -ne 0 ] || \
	     ! printf '%s\n' "$$output" | awk '$$0 == "PASS" { pass=1 } END { exit !pass }'; then \
	    return 1; \
	  fi; \
	}; \
	set -e; \
	run_case $(ROWS) $(DIM) $(REPEAT) bf16 none none; \
	run_case $(ROWS) $(DIM) $(REPEAT) fp16 none fp16; \
	run_case $(ROWS) $(DIM) $(REPEAT) bf16 fp8 none; \
	run_case $(ROWS) $(DIM) $(REPEAT) bf16 mxfp4 none; \
	run_case $(ROWS) $(DIM) $(REPEAT) mxfp8 fp8 bf16
