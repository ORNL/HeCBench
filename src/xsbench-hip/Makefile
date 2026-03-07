#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = Main.cu \
         Simulation.cu\
         io.cu \
         GridInit.cu \
         Materials.cu \
         XSutils.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../xsbench-cuda

# Linker Flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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

Main.o: ../xsbench-cuda/Main.cu ../xsbench-cuda/XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

Simulation.o: Simulation.cu ../xsbench-cuda/XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

io.o: ../xsbench-cuda/io.cu ../xsbench-cuda/XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

GridInit.o: ../xsbench-cuda/GridInit.cu ../xsbench-cuda/XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

Materials.o: ../xsbench-cuda/Materials.cu ../xsbench-cuda/XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

XSutils.o: ../xsbench-cuda/XSutils.cu ../xsbench-cuda/XSbench_header.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -s large -m event -r 10
