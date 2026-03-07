#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main vj-cpu

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -G
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

HEADERS := image.h haar.h stdio-wrapper.h

all: build

build: $(program)

image.o: image.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

stdio-wrapper.o: stdio-wrapper.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

main.o: main.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

haar-gpu.o: haar.cu $(HEADERS)
	$(CC) $(CFLAGS) -DGPU -o $@ -c $<

haar-cpu.o: haar.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

rectangles.o: rectangles.cu $(HEADERS)
	$(CC) $(CFLAGS) -o $@ -c $<

vj-cpu: main.o haar-cpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

main: main.o haar-gpu.o image.o stdio-wrapper.o rectangles.o
	$(CC) $(CFLAGS) -o $@ $+ $(LDFLAGS)

clean:
	rm -f $(program) *.o Output-cpu.pgm Output-gpu.pgm

run: build
	$(LAUNCHER) ./vj-cpu Face.pgm info.txt class.txt Output-cpu.pgm
	$(LAUNCHER) ./main Face.pgm info.txt class.txt Output-gpu.pgm
