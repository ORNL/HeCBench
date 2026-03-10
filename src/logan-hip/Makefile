#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = src/main.cu src/seed.cu src/score.cu src/logan_functions.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 \
          -Wall -fopenmp -Isrc 

# Linker Flags
LDFLAGS = -fgpu-rdc --hip-link 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -DADAPTABLE
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

src/main.o: src/main.cu src/logan.cuh Makefile
	$(CC) $(CFLAGS) -c -fgpu-rdc $< -o $@

src/seed.o: src/seed.cu src/seed.cuh Makefile
	$(CC) $(CFLAGS) -c -fgpu-rdc $< -o $@

src/score.o: src/score.cu src/score.cuh Makefile
	$(CC) $(CFLAGS) -c -fgpu-rdc $< -o $@

src/logan_functions.o: src/logan_functions.cu \
                       src/logan_functions.cuh src/seed.cuh Makefile
	$(CC) $(CFLAGS) -c -fgpu-rdc $< -o $@

clean:
	rm -rf $(program) $(obj)

demo: $(program)
	./$(program) ../logan-cuda/inputs/example.txt 17 21 1

run: $(program)
	$(LAUNCHER) ./$(program) ../logan-cuda/inputs/100k.txt 17 100 1
