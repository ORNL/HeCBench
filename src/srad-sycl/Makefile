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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl --gcc-toolchain=$(GCC_TOOLCHAIN)

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

./main.o:./main.cpp ./main.h \
	kernel_compress.sycl \
	kernel_extract.sycl \
	kernel_prepare.sycl \
	kernel_reduce.sycl \
	kernel_srad.sycl \
	kernel_srad2.sycl 
	$(CC) $(CFLAGS) $(KERNEL_DIM) ./main.cpp \
		-c \
		-o ./main.o \
		-O3

./util/graphics/graphics.o:./util/graphics/graphics.h ./util/graphics/graphics.c
	$(CC)	./util/graphics/graphics.c \
		-c \
		-o ./util/graphics/graphics.o \
		-O3

./util/graphics/resize.o:./util/graphics/resize.h ./util/graphics/resize.c
	$(CC)	./util/graphics/resize.c \
		-c \
		-o ./util/graphics/resize.o \
		-O3

./util/timer/timer.o:./util/timer/timer.h ./util/timer/timer.c
	$(CC)	./util/timer/timer.c \
		-c \
		-o ./util/timer/timer.o \
		-O3

clean:
	rm -f ./*.o main *.pgm ./util/graphics/*.o ./util/timer/*.o

run: main
	$(LAUNCHER) ./main 1000 0.5 502 458
