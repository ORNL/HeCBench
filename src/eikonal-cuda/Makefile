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

source = kernel.cu fim.cu timer.cu main.cpp StructuredEikonal.cu

obj = kernel.o fim.o timer.o main.o StructuredEikonal.o

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

%.o: %.cu Makefile
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

kernel.o: kernel.cu kernel.h
fim.o: fim.cu fim.h kernel.h
timer.o : timer.cpp timer.h my_exception.h
StructuredEikonal.o : StructuredEikonal.cu StructuredEikonal.h common_def.h
main.o : main.cpp StructuredEikonal.h

clean:
	rm -rf $(program) $(obj) output.nrrd

run: $(program)
	$(LAUNCHER) ./$(program) -s 512
