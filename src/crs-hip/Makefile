#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
OUTPUT    = no
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu \
         utils.cu \
         ../crs-cuda/galois.cu \
         ../crs-cuda/jerasure.cu \
         ../crs-cuda/GCRSMatrix.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../crs-cuda

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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): jerasure.o galois.o GCRSMatrix.o main.o utils.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

main.o: main.cu kernels.cu utils.h ../crs-cuda/GCRSMatrix.h 
	$(CC) $(CFLAGS) -c $< -o $@

utils.o: utils.cu utils.h
	$(CC) $(CFLAGS) -c $< -o $@

jerasure.o: ../crs-cuda/jerasure.cu ../crs-cuda/jerasure.h ../crs-cuda/galois.h
	$(CC) $(CFLAGS) -c $< -o $@

galois.o: ../crs-cuda/galois.cu ../crs-cuda/galois.h
	$(CC) $(CFLAGS) -c $< -o $@

GCRSMatrix.o: ../crs-cuda/GCRSMatrix.cu ../crs-cuda/GCRSMatrix.h ../crs-cuda/jerasure.h ../crs-cuda/galois.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) *.o

run: $(program)
	$(LAUNCHER) ./$(program) 10 10


