#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

c_source = ../prna-cuda/main.c ../prna-cuda/base.c ../prna-cuda/param.c ../prna-cuda/util.c
cuda_source = prna.cu 

obj = main.o base.o param.o util.o prna.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DFLOAT -std=c++17 -x hip -D__HIP_PLATFORM_AMD__ -Wall -I. -I../prna-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -ffast-math
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: ../prna-cuda/%.c
	$(CC) $(CFLAGS) -c $< -o $@

base.o :  ../prna-cuda/base.c  ../prna-cuda/base.h ../prna-cuda/util.h
main.o :  ../prna-cuda/main.c  ../prna-cuda/prna.h ../prna-cuda/util.h ../prna-cuda/base.h
param.o : ../prna-cuda/param.c ../prna-cuda/param.h ../prna-cuda/util.h ../prna-cuda/real.h
prna.o :  ../prna-cuda/prna.cu ../prna-cuda/base.h ../prna-cuda/prna.h ../prna-cuda/util.h \
	  ../prna-cuda/param.h ../prna-cuda/real.h cu.h
util.o :  ../prna-cuda/util.c ../prna-cuda/util.h

clean:
	rm -rf $(program) $(obj)

test: $(program)
	./$(program) ../prna-cuda/test.seq

run: $(program)
	$(LAUNCHER) ./$(program) ../prna-cuda/HIV1-NL43.seq
