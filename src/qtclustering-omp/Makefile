#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
LAUNCHER  =


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp Option.cpp OptionParser.cpp QTC.cpp comm.cpp libdata.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

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

ifeq ($(DEVICE),gpu)
  #CFLAGS +=-qnextgen -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp OptionParser.h Utility.h 
	$(CC) $(CFLAGS) -c $< -o $@

comm.o: comm.cpp comm.h
	$(CC) $(CFLAGS) -c $< -o $@

libdata.o: libdata.cpp libdata.h
	$(CC) $(CFLAGS) -c $< -o $@

Option.o: Option.cpp Option.h 
	$(CC) $(CFLAGS) -c $< -o $@

OptionParser.o: OptionParser.cpp OptionParser.h Utility.h
	$(CC) $(CFLAGS) -c $< -o $@

QTC.o: QTC.cpp QTC.h OptionParser.h libdata.h \
       tuningParameters.h comm.h kernels_compact_storage.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --Verbose
	#./$(program) --SaveOutput
