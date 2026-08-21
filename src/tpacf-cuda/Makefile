#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  ?=

POINTS    = 97178
DATAPARS  = -d ./data/small/Datapnts.1 -p $(POINTS)
RANDPARS  = -r ./data/small/Randompnts -n 100 -q $(POINTS)
ANALPARS  = -b 5 -l 1 -u 10000 -a -j 10
OUTFILE   = ./small_results_test.dat

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu args.cu compute.cu

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
  CFLAGS += -g -DDEBUG
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

main.o: main.cu args.h kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

args.o: args.cu args.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

compute.o: compute.cu model_io.cu ACF_kernel.cu histogram_kernel.cu args.h \
           kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) $(OUTFILE)

run: $(program)
	$(LAUNCHER) ./$(program) $(DATAPARS) $(RANDPARS) $(ANALPARS) -o $(OUTFILE)
