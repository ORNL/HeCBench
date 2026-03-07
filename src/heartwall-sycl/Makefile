#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
LAUNCHER=

OPTIMIZE  = yes
DEBUG     = no
DUMP      = yes

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

obj = $(source:.cpp=.o)

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
  TEST_ON = -DTEST_CHECKSUM 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(GPU),yes)
  CFLAGS +=-DUSE_GPU
endif

ifeq ($(DUMP),yes)
  OUTPUT  = -DOUTPUT
endif
#===============================================================================
# Targets to Build
#===============================================================================

main:     ./main.o \
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

./main.o:./main.h ./main.c
	$(CC)	$(OUTPUT) $(KERNEL_DIM) ./main.c \
		-I./util/timer/ \
		-I./util/file/ \
		-c \
		-o ./main.o 

./kernel/kernel.o:./kernel/kernel.sycl ./kernel/kernel.cpp
	$(CC) $(CFLAGS) $(TEST_ON) $(KERNEL_DIM) ./kernel/kernel.cpp \
		-c \
		-o ./kernel/kernel.o 

./util/avi/avilib.o:./util/avi/avilib.h ./util/avi/avilib.c
	$(CC)	$(CFLAGS) ./util/avi/avilib.c \
		-I./util/avi \
		-c \
		-o ./util/avi/avilib.o 

./util/avi/avimod.o:./util/avi/avilib.h ./util/avi/avimod.c
	$(CC)	$(CFLAGS) ./util/avi/avimod.c \
		-I./util/avi \
		-c \
		-o ./util/avi/avimod.o 

./util/file/file.o:./util/file/file.h ./util/file/file.c
	$(CC)	$(CFLAGS) ./util/file/file.c \
		-I./util/file \
		-I./util/timer \
		-c \
		-o ./util/file/file.o 

./util/timer/timer.o:./util/timer/timer.h ./util/timer/timer.c
	$(CC)	$(CFLAGS) ./util/timer/timer.c \
		-I./util/timer \
		-c \
		-o ./util/timer/timer.o 

clean:
	rm	-rf *.o main \
		./kernel/*.o \
		./util/avi/*.o \
		./util/file/*.o \
		./util/timer/*.o 

run: main
	$(LAUNCHER) ./main 104
