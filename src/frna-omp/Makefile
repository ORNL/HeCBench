#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

c_source = main.c fbase.c fparam.c util.c
omp_source = frna.cpp

obj = $(c_source:.c=.o) $(omp_source:.cpp=.o) 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DINT -std=c++17 -Wall

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-DOMP_TARGET -ffast-math -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
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
	./$(program) ../frna-cuda/RD0260.seq rd0260.out

run: $(program)
	$(LAUNCHER) ./$(program) ../prna-cuda/HIV1-NL43.seq  hiv1-nl43.out
