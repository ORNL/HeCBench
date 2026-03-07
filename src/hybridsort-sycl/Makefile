#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
LAUNCHER=

OPTIMIZE  = yes
DEBUG     = no
CHECK     = yes
DUMP      = yes
TIMING    = yes

GPU       = yes
CUDA      = no
CUDA_ARCH = sm_70
HIP       = no
HIP_ARCH  = gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

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

ifeq ($(VENDOR), codeplay)
LDFLAGS += -lComputeCpp
endif

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

ifeq ($(CHECK),yes)
  VERIFY =-DVERIFY
endif

ifeq ($(DUMP),yes)
  OUTPUT =-DOUTPUT
endif

ifeq ($(TIMING),yes)
  TIMER =-DTIMER
endif
#===============================================================================
# Targets to Build
#===============================================================================

main: hybridsort.o bucketsort.o mergesort.o
	$(CC) $(CFLAGS) -o main $(VERIFY) $(OUTPUT) $(TIMER) \
		hybridsort.o bucketsort.o mergesort.o $(LDFLAGS)

hybridsort.o : hybridsort.c mergesort.h bucketsort.h
	$(CC) $(CFLAGS) -o hybridsort.o -c hybridsort.c

bucketsort.o: bucketsort.c \
	kernel_bucketprefix.sycl \
	kernel_histogram.sycl \
	kernel_bucketcount.sycl \
       	kernel_bucketsort.sycl  
	$(CC) $(CFLAGS) -o bucketsort.o -c bucketsort.c 

mergesort.o: mergesort.c mergesort.h kernel_mergeSortPass.sycl
	$(CC) $(CFLAGS) -o mergesort.o -c mergesort.c 

run: main
	$(LAUNCHER) ./main r

clean:
	rm -f  *.o main
