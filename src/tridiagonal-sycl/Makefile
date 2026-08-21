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

program = main

source = main.cpp shrUtils.cpp cmd_arg_reader.cpp 

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -DUNIX -DBENCH_ITERATIONS=100 -std=c++17 -Wall -fsycl \
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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o : main.cpp \
         tridiagonal.h\
         shrUtils.h\
         file_read_write.h\
         test_gen_result_check.h\
         cpu_solvers.h\
         pcr_small_systems.h\
         cyclic_small_systems.h\
         sweep_small_systems.h\
         pcr_kernels.cpp\
         cyclic_kernels.cpp\
         sweep_kernels.cpp
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o: shrUtils.cpp shrUtils.h cmd_arg_reader.h exception.h
	$(CC) $(CFLAGS) -c $< -o $@

cmd_arg_reader.o: cmd_arg_reader.cpp cmd_arg_reader.h exception.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.dat *.timing oclTridiagonal.txt SdkConsoleLog.txt *.csv

run: $(program)
	$(LAUNCHER) ./$(program) -num_systems=524288
	$(LAUNCHER) ./$(program) --lmem -num_systems=524288
	$(LAUNCHER) ./$(program) --vec4 -num_systems=524288 
