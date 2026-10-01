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

source = kernel.cu fim.cu StructuredEikonal.cu \
         timer.cpp main.cpp 

obj = kernel.o fim.o StructuredEikonal.o \
      timer.o main.o 

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../eikonal-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

kernel.o: kernel.cu kernel.h
	$(CC) $(CFLAGS) -c $< -o $@

fim.o: fim.cu fim.h kernel.h
	$(CC) $(CFLAGS) -c $< -o $@

StructuredEikonal.o : StructuredEikonal.cu ../eikonal-cuda/StructuredEikonal.h \
                      ../eikonal-cuda/common_def.h
	$(CC) $(CFLAGS) -c $< -o $@

timer.o : ../eikonal-cuda/timer.cpp ../eikonal-cuda/timer.h ../eikonal-cuda/my_exception.h
	$(CC) $(CFLAGS) -c $< -o $@

main.o : ../eikonal-cuda/main.cpp ../eikonal-cuda/StructuredEikonal.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output.nrrd

run: $(program)
	$(LAUNCHER) ./$(program) -s 512
