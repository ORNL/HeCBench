CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =


# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g  
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif


./main:	./main.o \
	./util/graphics/graphics.o \
	./util/graphics/resize.o \
	./util/timer/timer.o
	$(CC) $(CFLAGS)	./main.o \
		./util/graphics/graphics.o \
		./util/graphics/resize.o \
		./util/timer/timer.o \
		-lm \
		$(LDFLAGS) -o main

./main.o: ./main.c ./main.h
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./main.c \
		-c \
		-o ./main.o \
		-O3

./util/graphics/graphics.o:	./util/graphics/graphics.h \
	./util/graphics/graphics.c
	$(CC)	./util/graphics/graphics.c \
		-c \
		-o ./util/graphics/graphics.o \
		-O3

./util/graphics/resize.o:	./util/graphics/resize.h \
	./util/graphics/resize.c
	$(CC)	./util/graphics/resize.c \
		-c \
		-o ./util/graphics/resize.o \
		-O3

./util/timer/timer.o:	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

clean:
	rm -f ./*.o main *.pgm ./util/graphics/*.o ./util/timer/*.o

run: main
	$(LAUNCHER) ./main 1000 0.5 502 458
