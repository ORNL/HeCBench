CC          = icpx
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu
LAUNCHER    =



# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
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


ifdef VERIFY
override VERIFY = -DVERIFY
endif

ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

ifdef TIMER
override TIMER = -DTIMER
endif

main: hybridsort.o bucketsort.o mergesort.o
	$(CC) $(CFLAGS) -o main $(VERIFY) $(OUTPUT) $(TIMER) \
		hybridsort.o \
		bucketsort.o \
                mergesort.o \
		$(LDFLAGS)

hybridsort.o: hybridsort.c bucketsort.h mergesort.h
	$(CC) $(CFLAGS) -o hybridsort.o -c hybridsort.c 

bucketsort.o: bucketsort.c \
	kernel_bucketprefix.h \
	kernel_histogram.h \
	kernel_bucketcount.h \
       	kernel_bucketsort.h  
	$(CC) $(CFLAGS) -o bucketsort.o -c bucketsort.c 

mergesort.o: mergesort.c mergesort.h kernel_mergeSortPass.h
	$(CC) $(CFLAGS) -o mergesort.o -c mergesort.c 

run: main
	$(LAUNCHER) ./main r

clean:
	rm -f  *.o main
