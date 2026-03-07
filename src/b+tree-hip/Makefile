#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif


# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3

endif

main:	./main.o \
		./kernel/kernel_wrapper.o \
		./kernel/kernel2_wrapper.o \
		./util/timer/timer.o \
		./util/num/num.o 
	$(CC)	$(CFLAGS) ./main.o \
			./kernel/kernel_wrapper.o \
			./kernel/kernel2_wrapper.o \
			./util/timer/timer.o \
			./util/num/num.o \
			-lm  -o main



main.o:	./common.h \
		./main.h \
		./main.cu
	$(CC)	$(CFLAGS) $(KERNEL_DIM) ./main.cu \
			-c \
			-o ./main.o

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_wrapper.o:	./common.h \
									./kernel/kernel_wrapper.h \
									./kernel/kernel_wrapper.cu
	$(CC) $(KERNEL_DIM) ./kernel/kernel_wrapper.cu \
				-c \
				-o ./kernel/kernel_wrapper.o \
				$(CFLAGS)

./kernel/kernel2_wrapper.o:	./common.h \
										./kernel/kernel2_wrapper.h \
										./kernel/kernel2_wrapper.cu
	$(CC) $(KERNEL_DIM) ./kernel/kernel2_wrapper.cu \
				-c \
				-o ./kernel/kernel2_wrapper.o \
				$(CFLAGS)

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/timer/timer.o:	./common.h \
						./util/timer/timer.h \
						./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
			-c \
			-o ./util/timer/timer.o 

./util/num/num.o:	./common.h \
					./util/num/num.h \
					./util/num/num.c
	$(CC)	./util/num/num.c \
			-c \
			-o ./util/num/num.o 

run: main
	$(LAUNCHER) ./main file ../data/b+tree/mil.txt command ../data/b+tree/command.txt

clean:
	rm	-rf *.o *.out \
		./kernel/*.o \
		./util/timer/*.o \
		./util/num/*.o \
    output.txt
