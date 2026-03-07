#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main vj-cpu

#===============================================================================
# Sets Flags
#===============================================================================

OMP_TARGET_FLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -Wall -I../face-sycl

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

HEADERS := haar.h ../face-sycl/image.h ../face-sycl/stdio-wrapper.h

all: build

build: $(program)

image.o: ../face-sycl/image.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

stdio-wrapper.o: ../face-sycl/stdio-wrapper.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

main.o: ../face-sycl/main.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

haar-gpu.o: haar.cpp $(HEADERS)
	$(CC) $(CFLAGS) -DOMP_TARGET $(OMP_TARGET_FLAGS) -o $@ -c $<

haar-cpu.o: haar.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

rectangles.o: ../face-sycl/rectangles.cpp $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

vj-cpu: main.o haar-cpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ 

main: main.o haar-gpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) $(OMP_TARGET_FLAGS) -o $@ $+ 

clean:
	rm -f $(program) *.o Output-cpu.pgm Output-gpu.pgm

run: build
	$(LAUNCHER) ./vj-cpu ../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-cpu.pgm
	$(LAUNCHER) ./main ../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-gpu.pgm
