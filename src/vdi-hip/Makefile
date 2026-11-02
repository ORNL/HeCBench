# HIP port of vdi. Wave size is the device wavefront (64 on MI210).
# No fast-math: the factorization and the inverse products stay in fp32.

CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

FRAMES ?= 101
HEADS  ?= 7
TOKENS ?= 1008
DIM    ?= 128
REPEAT ?= 100

program = main
source = main.cu fused.cu libraries.cu
obj = $(source:.cu=.o)

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../vdi-cuda
LDFLAGS = -lhipblas -lhipsolver -lpthread

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

main.o: ../vdi-cuda/host_common.hpp ../vdi-cuda/reference.h fused.h libraries.h
fused.o: fused.h ../vdi-cuda/local_layout.hpp
libraries.o: libraries.h ../vdi-cuda/library_batch.cuh

clean:
	rm -rf $(program) $(obj)

.PHONY: clean run

run: $(program)
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 1
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 50
	$(LAUNCHER) ./$(program) --frames $(FRAMES) --heads $(HEADS) --tokens $(TOKENS) --d $(DIM) --repeat $(REPEAT) --beta-scale 100
