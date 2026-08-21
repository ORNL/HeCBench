#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu  config.h kernels.h reference.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --op bgmv_shrink --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_shrink --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --vectorize --tokens 1 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
	$(LAUNCHER) ./$(program) --op bgmv_expand --vectorize --tokens 512 --hidden 4096 --rank 16 --loras 4 --repeat 200
