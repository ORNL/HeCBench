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

source = main.cpp \
../cmp-cuda/reference.cpp \
../cmp-cuda/su_cdp.cpp \
../cmp-cuda/su_gather.cpp \
../cmp-cuda/su_trace.cpp \
../cmp-cuda/log.cpp  \
../cmp-cuda/parser.cpp  

obj = main.o reference.o su_cdp.o su_gather.o su_trace.o log.o parser.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../cmp-cuda

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
  CFLAGS +=-fopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o : main.cpp ../cmp-cuda/log.hpp ../cmp-cuda/utils.hpp ../cmp-cuda/parser.hpp ../cmp-cuda/su_gather.hpp
	$(CC) $(CFLAGS) -c $< -o $@

%.o: ../cmp-cuda/%.cpp
	$(CC) $(CFLAGS) -c $< -o $@

log.o : ../cmp-cuda/log.cpp ../cmp-cuda/log.hpp
parser.o: ../cmp-cuda/parser.cpp ../cmp-cuda/parser.hpp  
su_cdp.o : ../cmp-cuda/su_cdp.cpp ../cmp-cuda/su_cdp.hpp ../cmp-cuda/log.hpp
su_gather.o : ../cmp-cuda/su_gather.cpp ../cmp-cuda/su_gather.hpp ../cmp-cuda/log.hpp ../cmp-cuda/su_trace.hpp
su_trace.o : ../cmp-cuda/su_trace.cpp ../cmp-cuda/su_trace.hpp ../cmp-cuda/log.hpp
reference.o : ../cmp-cuda/reference.cpp ../cmp-cuda/utils.hpp

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -c0 1.98e-7 -c1 1.77e-6 -nc 5 -aph 600 -tau 0.002 -v 2 \
                     -i ../cmp-cuda/data/simple-synthetic.su
