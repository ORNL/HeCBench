#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =



# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
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

all: main

main: cluster.o getopt.o read_input.o rmse.o kmeans.c
	$(CC) $(KERNEL_DIM) $(CFLAGS) $^ -o main $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $< -c

kmeans.o: kmeans.c kmeans.h

cluster.o: cluster.c kmeans.h

getopt.o: getopt.c getopt.h

read_input.o: read_input.c kmeans.h

rmse.o: rmse.c kmeans.h

clean:
	rm -f *.o *~ main *.linkinfo

run: main
	$(LAUNCHER) ./main -r -n 5 -m 15 -l 10 -o -i ../data/kmeans/kdd_cup


