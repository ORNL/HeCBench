#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
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
CFLAGS := $(EXTRA_CFLAGS) -std=c++14 \
          -Xcompiler -Wall -Xcompiler -fopenmp \
          -arch=$(ARCH) -Isrc 

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3 -maxrregcount=32 -DADAPTABLE
endif

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

src/main.o: src/main.cu src/logan.cuh Makefile
	$(CC) $(CFLAGS) -c -dc $< -o $@

src/seed.o: src/seed.cu src/seed.cuh Makefile
	$(CC) $(CFLAGS) -c -dc $< -o $@

src/score.o: src/score.cu src/score.cuh Makefile
	$(CC) $(CFLAGS) -c -dc $< -o $@

src/logan_functions.o: src/logan_functions.cu \
                       src/logan_functions.cuh src/seed.cuh Makefile
	$(CC) $(CFLAGS) -c -dc $< -o $@

clean:
	rm -rf $(program) $(obj)

demo: $(program)
	./$(program) inputs/example.txt 17 21 1

run: $(program)
	$(LAUNCHER) ./$(program) inputs/100k.txt 17 100 1
