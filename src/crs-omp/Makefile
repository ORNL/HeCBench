#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
OUTPUT    = no
DEVICE    = gpu
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp \
         utils.cpp \
         ../crs-sycl/galois.cpp \
         ../crs-sycl/jerasure.cpp \
         ../crs-sycl/GCRSMatrix.cpp

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -Wall -I../crs-sycl

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g 
endif

# Dump output
ifeq ($(OUTPUT),yes)
  CFLAGS += -DDUMP
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): jerasure.o galois.o GCRSMatrix.o main.o utils.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cpp kernels.cpp utils.h ../crs-sycl/GCRSMatrix.h 
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.cpp utils.h
	$(CC) $(CFLAGS) -c $< -o $@

jerasure.o: ../crs-sycl/jerasure.cpp ../crs-sycl/jerasure.h ../crs-sycl/galois.h
	$(CC) $(CFLAGS) -c $< -o $@

galois.o: ../crs-sycl/galois.cpp ../crs-sycl/galois.h
	$(CC) $(CFLAGS) -c $< -o $@

GCRSMatrix.o: ../crs-sycl/GCRSMatrix.cpp ../crs-sycl/GCRSMatrix.h ../crs-sycl/jerasure.h ../crs-sycl/galois.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) *.o

run: $(program)
	$(LAUNCHER) ./$(program) 10 10
