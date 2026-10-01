#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  ?=

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
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../bgmv-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g
  LDFLAGS  += -g
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
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp ../bgmv-cuda/reference.h ../bgmv-cuda/config.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --op bgmv_shrink --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
