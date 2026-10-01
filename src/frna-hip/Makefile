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

c_source = ../frna-cuda/main.c ../frna-cuda/fbase.c ../frna-cuda/fparam.c ../frna-cuda/util.c
cuda_source = frna.cu 

obj = $(c_source:.c=.o) $(cuda_source:.cu=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DINT -std=c++17 -x hip -D__HIP_PLATFORM_AMD__ -Wall -I. -I../frna-cuda

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

%.o: ../frna-cuda/%.c
	$(CC) $(CFLAGS) -c $< -o $@

fbase.o : ../frna-cuda/fbase.c ../frna-cuda/fbase.h ../frna-cuda/util.h cu.h
main.o : ../frna-cuda/main.c ../frna-cuda/frna.h ../frna-cuda/util.h ../frna-cuda/fbase.h
fparam.o : ../frna-cuda/fparam.c ../frna-cuda/fparam.h ../frna-cuda/util.h ../frna-cuda/fbase.h ../frna-cuda/int.h
fprna.o : ../frna-cuda/fprna.cu ../frna-cuda/frna.h ../frna-cuda/util.h cu.h
util.o : ../frna-cuda/util.c ../frna-cuda/util.h

clean:
	rm -rf $(program) $(obj) *.out

test: $(program)
	./$(program) ../frna-cuda/RD0260.seq rd0260.out

run: $(program)
	$(LAUNCHER) ./$(program) ../prna-cuda/HIV1-NL43.seq  hiv1-nl43.out
