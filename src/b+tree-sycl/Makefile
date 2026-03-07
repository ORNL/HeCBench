#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
LAUNCHER=

OPTIMIZE  = yes
DEBUG     = no

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================


#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -fsycl -Wall \
          -I. -I./util/num -I./util/timer -I./kernel \
          --gcc-toolchain=$(GCC_TOOLCHAIN)

ifeq ($(VENDOR), AdaptiveCpp)
    CFLAGS_TMP := $(CFLAGS)
    CFLAGS = $(filter-out -fsycl, $(CFLAGS_TMP))
endif

# Linker Flags
LDFLAGS = 

ifeq ($(CUDA), yes)
  CFLAGS += -fsycl-targets=nvptx64-nvidia-cuda \
            -Xsycl-target-backend --cuda-gpu-arch=$(CUDA_ARCH)
endif

ifeq ($(HIP), yes)
  CFLAGS += -fsycl-targets=amdgcn-amd-amdhsa \
	    -Xsycl-target-backend --offload-arch=$(HIP_ARCH) 
endif

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS  += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif
#===============================================================================
# Targets to Build
#===============================================================================

main: ./main.o \
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


main.o:	./b+tree.h ./main.c
	$(CC) $(CFLAGS) $(KERNEL_DIM) ./main.c -c -o main.o


./kernel/kernel_wrapper.o: ./b+tree.h \
                    ./kernel/kernel_wrapper.h \
                    ./kernel/kernel_wrapper.c \
                    ./kernel/findK.sycl
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel_wrapper.c \
		-c \
		-o ./kernel/kernel_wrapper.o

./kernel/kernel2_wrapper.o: ./b+tree.h \
                   ./kernel/kernel2_wrapper.h \
                   ./kernel/kernel2_wrapper.c \
                   ./kernel/findRangeK.sycl
	$(CC) $(CFLAGS)	$(KERNEL_DIM) ./kernel/kernel2_wrapper.c \
		-c \
		-o ./kernel/kernel2_wrapper.o 


./util/timer/timer.o: ./b+tree.h \
                  ./util/timer/timer.h \
                  ./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

./util/num/num.o: ./b+tree.h \
                  ./util/num/num.h \
                  ./util/num/num.c
	$(CC)	./util/num/num.c \
		-c \
		-o ./util/num/num.o \
		-O3

run: main
	$(LAUNCHER) ./main file ../data/b+tree/mil.txt command ../data/b+tree/command.txt

clean:
	rm -f	*.o main \
		./kernel/*.o \
		./util/timer/*.o \
		./util/num/*.o \
		output.txt
