#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

POINTS    = 97178
DATAPARS  = -d ../tpacf-cuda/data/small/Datapnts.1 -p $(POINTS)
RANDPARS  = -r ../tpacf-cuda/data/small/Randompnts -n 100 -q $(POINTS)
ANALPARS  = -b 5 -l 1 -u 10000 -a -j 10
OUTFILE   = ./small_results_test.dat

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

obj = main.o args.o compute.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../tpacf-cuda

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

main.o: ../tpacf-cuda/main.cu ../tpacf-cuda/args.h ../tpacf-cuda/kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

args.o: ../tpacf-cuda/args.cu ../tpacf-cuda/args.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

compute.o: compute.cu ../tpacf-cuda/args.h ../tpacf-cuda/model_io.cu \
           ACF_kernel.cu histogram_kernel.cu ../tpacf-cuda/kernel.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) $(OUTFILE)

run: $(program)
	$(LAUNCHER) ./$(program) $(DATAPARS) $(RANDPARS) $(ANALPARS) -o $(OUTFILE)
