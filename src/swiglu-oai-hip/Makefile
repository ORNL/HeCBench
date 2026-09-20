#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

ROWS   ?= 8192
DIM    ?= 1024
REPEAT ?= 100

# E8M0 conversion API selection:
#   auto - use HIP version detection in main.cu (default)
#   yes  - force __hip_cvt_float_to_e8m0 (for vendor backports)
#   no   - force the portable exponent-bit fallback
HIP_E8M0_CONVERSION ?= auto

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

ifeq ($(HIP_E8M0_CONVERSION),yes)
  CFLAGS += -DSWIGLU_HAS_HIP_E8M0_CONVERSION=1
else ifeq ($(HIP_E8M0_CONVERSION),no)
  CFLAGS += -DSWIGLU_HAS_HIP_E8M0_CONVERSION=0
else ifneq ($(HIP_E8M0_CONVERSION),auto)
  $(error HIP_E8M0_CONVERSION must be auto, yes, or no)
endif

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

%.o: %.cu reference.h ../swiglu-oai-cuda/reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

.PHONY: clean run

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
