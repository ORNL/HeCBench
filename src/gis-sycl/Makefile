#===============================================================================
# User Options
#===============================================================================

CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908

# FBGEMM group_index_select_2d_bench production shape (pytorch/FBGEMM#5487):
# 3 tables, 288841 rows, embedding dimension 96, and 1019100 indices.
TABLES   ?= 3
ROWS     ?= 288841
DIM      ?= 96
SELECTED ?= 339700
REPEAT   ?= 100
# Synthetic cap on repeated index runs, not a production measurement.
# FBGEMM sorts backward indices, so a production run equals the number of
# lookups that selected the same row.
MAX_RUN  ?= 8

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../gis-cuda -fsycl

ifneq ($(GCC_TOOLCHAIN),)
  CFLAGS += --gcc-toolchain=$(GCC_TOOLCHAIN)
endif

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

LDFLAGS =

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
            -Xsycl-target-backend --offload-arch=$(HIP_ARCH)
endif

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS += -DUSE_GPU
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../gis-cuda/gis_common.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(TABLES) $(ROWS) $(DIM) $(SELECTED) $(REPEAT) $(MAX_RUN)

.PHONY: clean run
