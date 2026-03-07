#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC          = hipcc
OPTIMIZE    = yes
DEBUG       = no
LAUNCHER    =

#===============================================================================
# Program name & source code list
#===============================================================================
program = main

source = main.cu

obj = $(source:.cu=.o)


#===============================================================================
# Sets Flags
#===============================================================================

# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker flags
LDFLAGS = -lm

# Debug Flags
ifeq ($(DEBUG),yes)
	CFLAGS += -g  
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

%.o: %.cu graphics.c resize.c timer.c \
	extract_kernel.cu prepare_kernel.cu \
	reduce_kernel.cu srad_kernel.cu srad2_kernel.cu compress_kernel.cu
	$(CC) $(CFLAGS) -c $< -o $@

# delete all object files
clean:
	rm -f *.o $(program) *pgm

run: $(program)
	$(LAUNCHER) ./$(program) 1000 0.5 502 458
