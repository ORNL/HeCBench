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
# Matrix library path
MATRIX_DIR = ./meschach_lib

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I$(MATRIX_DIR) -fsycl \
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
  OUTPUT = -DOUTPUT
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

main: detect_main.o \
	avilib.o \
	track_ellipse.o \
	track_ellipse_sycl.o \
	find_ellipse.o \
	misc_math.o \
	helper.o \
	$(MATRIX_DIR)/meschach.a
	$(CC) $(CFLAGS) -o main avilib.o \
		track_ellipse.o \
		track_ellipse_sycl.o \
		find_ellipse.o \
		misc_math.o \
		detect_main.o \
		helper.o \
		$(MATRIX_DIR)/meschach.a $(LDFLAGS)

%.o: %.[c]
	$(CC) $(OUTPUT) $(CFLAGS) $< -c

detect_main.o: detect_main.c \
               kernel_GICOV.sycl kernel_dilated.sycl \
               track_ellipse.h helper.h misc_math.h

find_ellipse.o: find_ellipse.c find_ellipse.h avilib.h misc_math.h

track_ellipse.o: track_ellipse.c find_ellipse.h $(MATRIX_DIR)/matrix.h avilib.h helper.h

track_ellipse_sycl.o: track_ellipse_sycl.c track_ellipse.h kernel_IMGVF.sycl


# Use the sycl compiler to build the library
# Cleanup everything, then clean everything except the static library
$(MATRIX_DIR)/meschach.a:
	cd $(MATRIX_DIR); make cleanup; make all CC=$(CC) CFLAGS=-O3; make clean

clean:
	rm -f *.o main
	cd $(MATRIX_DIR); make cleanup

run: main
	$(LAUNCHER) ./main ../data/leukocyte/testfile.avi 100
