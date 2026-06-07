#===============================================================================
# User Options
#===============================================================================

CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp flowOMP.cpp flowGold.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../opticalFlow-cuda/Common

LDFLAGS =

ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS += -g
endif

ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  # OpenMP 4.5 target offload (map/is_device_ptr; no omp_target_alloc)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -fopenmp-version=45 -D__STRICT_ANSI__
else
  CFLAGS += -qopenmp
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp *.h *.hpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.flo

run: $(program)
	$(LAUNCHER) ./$(program) ../opticalFlow-cuda/data/frame10.ppm ../opticalFlow-cuda/data/frame11.ppm
