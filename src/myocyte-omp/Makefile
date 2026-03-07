CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

# Standard flags
#CFLAGS := $(EXTRA_CFLAGS) -std=c++11 -Wall
CFLAGS := $(EXTRA_CFLAGS)  -Wall

# Linker flags
LDFLAGS = -lm

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g  -DDEBUG
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
	CFLAGS += -O3
endif


#	EXECUTABLES (LINK OBJECTS TOGETHER INTO BINARY)

./main:	./main.o \
	./kernel/kernel_wrapper.o \
	./util/file/file.o \
	./util/num/num.o
	$(CC) $(CFLAGS) ./main.o \
		./kernel/kernel_wrapper.o \
		./util/file/file.o \
		./util/num/num.o \
		$(LDFLAGS) \
		-o main

#	MAIN FUNCTION

./main.o:	./common.h ./main.c
	$(CC) $(CFLAGS) ./main.c \
		-c \
		-o ./main.o \
		-O3

# ======================================================================================================================================================150
#	KERNELS
# ======================================================================================================================================================150

./kernel/kernel_wrapper.o:	./common.h \
	./kernel/kernel_wrapper.h \
	./kernel/kernel_wrapper.c \
	./kernel/solver.c \
	./kernel/embedded_fehlberg_7_8.c \
	./kernel/master.c \
	./kernel/kernel_cam.h \
	./kernel/kernel_ecc.h
	$(CC) $(CFLAGS)	./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o 

./util/file/file.o:	./common.h \
	./util/file/file.h \
	./util/file/file.c
	$(CC)	./util/file/file.c \
		-c \
		-o ./util/file/file.o \
		-O3

./util/num/num.o:	./common.h \
	./util/num/num.h \
	./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

run: main
	$(LAUNCHER) ./main -time 100

clean:
	rm -rf	*.o main \
		./kernel/*.o \
		./util/file/*.o \
		./util/num/*.o \
		./output.txt
