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

program = main vj-cpu

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../face-cuda

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

HEADERS := ../face-cuda/haar.h ../face-cuda/image.h ../face-cuda/stdio-wrapper.h

all: build

build: $(program)

image.o: ../face-cuda/image.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

stdio-wrapper.o: ../face-cuda/stdio-wrapper.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

main.o: ../face-cuda/main.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

haar-gpu.o: haar.cu $(HEADERS)
	$(CC) $(CFLAGS) -DGPU -o $@ -c $<

haar-cpu.o: haar.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

rectangles.o: ../face-cuda/rectangles.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

vj-cpu: main.o haar-cpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

main: main.o haar-gpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -f $(program) *.o Output-cpu.pgm Output-gpu.pgm

run: build
	$(LAUNCHER) ./vj-cpu ../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-cpu.pgm
	$(LAUNCHER) ./main ../face-cuda/Face.pgm ../face-cuda/info.txt ../face-cuda/class.txt Output-gpu.pgm
