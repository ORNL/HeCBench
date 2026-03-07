#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
PROFILE   = no
MPI       = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = Main.o io.o Simulation.o GridInit.o XSutils.o Materials.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -Wall -I../xsbench-cuda
          
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

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

Main.o: ../xsbench-cuda/Main.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

Simulation.o: Simulation.cpp ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

io.o: ../xsbench-cuda/io.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

GridInit.o: ../xsbench-cuda/GridInit.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

Materials.o: ../xsbench-cuda/Materials.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

XSutils.o: ../xsbench-cuda/XSutils.cu ../xsbench-cuda/XSbench_header.h
	$(CC) -x c++ $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -s large -m event -r 10
