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

SPATH = ../boxfilter-sycl

program = main

obj = main.o shrUtils.o cmd_arg_reader.o reference.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../boxfilter-sycl -fPIE

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cu $(SPATH)/shrUtils.h 
	$(CC) $(CFLAGS) -c $< -o $@

shrUtils.o : $(SPATH)/shrUtils.cpp $(SPATH)/shrUtils.h $(SPATH)/cmd_arg_reader.h $(SPATH)/exception.h
	$(CC) $(CFLAGS) -c $< -o $@

cmd_arg_reader.o : $(SPATH)/cmd_arg_reader.cpp $(SPATH)/cmd_arg_reader.h $(SPATH)/exception.h
	$(CC) $(CFLAGS) -c $< -o $@

reference.o : $(SPATH)/reference.cpp 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../boxfilter-sycl/data/lenaRGB.ppm 10000
