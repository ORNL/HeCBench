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

c_source = main.c fbase.c fparam.c util.c
cuda_source = frna.cu 

obj = $(c_source:.c=.o) $(cuda_source:.cu=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DINT -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

fbase.o : fbase.c fbase.h util.h cu.h
main.o : main.c frna.h util.h fbase.h
fparam.o : fparam.c fparam.h util.h fbase.h int.h
fprna.o : fprna.cu frna.h util.h
util.o : util.c util.h

clean:
	rm -rf $(program) $(obj) *.out

test: $(program)
	./$(program) ./RD0260.seq rd0260.out

run: $(program)
	$(LAUNCHER) ./$(program) ../prna-cuda/HIV1-NL43.seq  hiv1-nl43.out
