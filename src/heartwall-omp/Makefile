CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DUMP      = yes
DEVICE    = gpu
LAUNCHER  =



# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++11 -Wall

# Linker flags
LDFLAGS = -lm

# Enable checksum and output file
ifeq ($(DEBUG),yes)
  CFLAGS += -g  
  TEST_ON = -DTEST_CHECKSUM 
endif

ifeq ($(DUMP),yes)
  OUTPUT  = -DOUTPUT
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
	    ./kernel/kernel.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o 
	$(CC)	$(CFLAGS) ./main.o \
		  ./kernel/kernel.o \
			./util/avi/avilib.o \
			./util/avi/avimod.o \
			./util/file/file.o \
			./util/timer/timer.o \
			$(LDFLAGS) -o main

# ======================================================================================================================================================150
#	MAIN FUNCTION
# ======================================================================================================================================================150

./main.o:	./main.cpp ./main.h
	$(CC)	$(OUTPUT) $(KERNEL_DIM) ./main.cpp \
		  -I./util/timer/ \
		  -I./util/file/ \
			-c \
			-o ./main.o 

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel.o:./kernel/kernel.cpp ./kernel/kernel.h
	$(CC) $(CFLAGS) $(TEST_ON) $(KERNEL_DIM)	./kernel/kernel.cpp \
			-c \
			-o ./kernel/kernel.o 

# ======================================================================================================================================================150
#	UTILITIES
# ======================================================================================================================================================150

./util/avi/avilib.o:./util/avi/avilib.h \
					./util/avi/avilib.c
	$(CC)	$(CFLAGS) ./util/avi/avilib.c \
		  -I./util/avi \
			-c \
			-o ./util/avi/avilib.o 

./util/avi/avimod.o:./util/avi/avilib.h \
					./util/avi/avimod.c
	$(CC)	$(CFLAGS) ./util/avi/avimod.c \
		  -I./util/avi \
			-c \
			-o ./util/avi/avimod.o 

./util/file/file.o:	./util/file/file.h \
					./util/file/file.c
	$(CC)	$(CFLAGS) ./util/file/file.c \
		  -I./util/file \
		  -I./util/timer \
			-c \
			-o ./util/file/file.o 

./util/timer/timer.o:	./util/timer/timer.h \
						./util/timer/timer.c
	$(CC)	$(CFLAGS) ./util/timer/timer.c \
		  -I./util/timer \
			-c \
			-o ./util/timer/timer.o 

run: main
	$(LAUNCHER) ./main 104

clean:
	rm	-rf *.o main \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o 
