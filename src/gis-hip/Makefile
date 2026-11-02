#===============================================================================
# User Options
#===============================================================================

CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      ?= gfx90a
LAUNCHER  ?=

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

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../gis-cuda --offload-arch=$(ARCH)

LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -munsafe-fp-atomics
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu ../gis-cuda/gis_common.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) $(TABLES) $(ROWS) $(DIM) $(SELECTED) $(REPEAT) $(MAX_RUN)

.PHONY: clean run
