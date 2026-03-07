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


main:	./main.o \
		./kernel/kernel_wrapper.o \
		./kernel/kernel2_wrapper.o \
		./util/timer/timer.o \
		./util/num/num.o 
	$(CC) $(CFLAGS)	./main.o \
			./kernel/kernel_wrapper.o \
			./kernel/kernel2_wrapper.o \
			./util/timer/timer.o \
			./util/num/num.o \
			$(LDFLAGS) \
	                -o main
# ========================================================================================================================================================================================================200
#	OBJECTS (COMPILE SOURCE FILES INTO OBJECTS)
# ========================================================================================================================================================================================================200

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

main.o:	./common.h \
	./main.c
	$(CC) $(CFLAGS) $(KERNEL_DIM) ./main.c \
		-c \
		-o ./main.o

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_wrapper.o:	./common.h \
	./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.c
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o

./kernel/kernel2_wrapper.o:	./common.h \
	./kernel/kernel2_wrapper.h \
	./kernel/kernel2_wrapper.c
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel2_wrapper.c \
		-c \
		-o ./kernel/kernel2_wrapper.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/timer/timer.o:	./common.h \
	./util/timer/timer.h \
	./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

./util/num/num.o:	./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

# ======================================================================================================================================================150
#	END
# ======================================================================================================================================================150
#
run: main
	$(LAUNCHER) ./main file ../data/b+tree/mil.txt command ../data/b+tree/command.txt

# ========================================================================================================================================================================================================200
#	DELETE
# ========================================================================================================================================================================================================200

clean:
	rm -f	*.o main \
		./kernel/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		output.txt
