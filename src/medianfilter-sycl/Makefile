#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = clang++
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

GPU       = yes
CUDA      = no
CUDA_ARCH ?= sm_70
HIP       = no
HIP_ARCH  ?= gfx908
#GCC_TOOLCHAIN = "/auto/software/gcc/x86_64/gcc-9.1.0/"

#===============================================================================
# Program name & source code list
#===============================================================================
UPATH = ../boxfilter-sycl
SPATH = ../medianfilter-cuda

program = main

obj = main.o shrUtils.o MedianFilterHost.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fsycl \
          -I$(UPATH) -I$(SPATH) --gcc-toolchain=$(GCC_TOOLCHAIN)

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp MedianFilter.cpp $(UPATH)/shrUtils.h
	$(CC) $(CFLAGS) -c $< -o $@

MedianFilterHost.o : $(SPATH)/MedianFilterHost.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o : $(UPATH)/shrUtils.cpp $(UPATH)/shrUtils.h 
	$(CC) $(CFLAGS) -c $< -o $@

# 2161 is not a multiple of the 4-row work group. The output buffer is sized
# to the image, so this run stores one row past the end if the restored Y
# bounds check is off by one.
LARGE_W = 3840
LARGE_H = 2161
LARGE_PPM = large-$(LARGE_W)x$(LARGE_H).ppm

$(LARGE_PPM):
	python3 -c 'w=$(LARGE_W); h=$(LARGE_H); p="$@"; row=bytes((x*17+c*3)&255 for x in range(w) for c in range(3)); f=open(p,"wb"); f.write(("P6\n%d %d\n255\n"%(w,h)).encode()); [f.write(bytes((b+y)&255 for b in row)) for y in range(h)]; f.close()'

clean:
	rm -rf $(program) $(obj) $(LARGE_PPM)

run: $(program) $(LARGE_PPM)
	$(LAUNCHER) ./$(program) ./data/SierrasRGB.ppm 1000
	$(LAUNCHER) ./$(program) $(LARGE_PPM) 100
