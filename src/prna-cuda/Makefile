#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

c_source = main.c base.c param.c util.c
cuda_source = prna.cu 

obj = $(c_source:.c=.o) $(cuda_source:.cu=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DFLOAT -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -use_fast_math
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

base.o : base.c base.h util.h
main.o : main.c prna.h util.h base.h
param.o : param.c param.h util.h  real.h
prna.o : prna.cu base.h prna.h util.h param.h real.h cu.h
util.o : util.c util.h

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) ./test.seq

run: $(program)
	$(LAUNCHER) ./$(program) ./HIV1-NL43.seq
