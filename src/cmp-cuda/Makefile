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

cuda_source = main.cu
cpp_source = reference.cpp su_cdp.cpp  su_gather.cpp  su_trace.cpp  log.cpp  parser.cpp  

obj = $(cuda_source:.cu=.o) $(cpp_source:.cpp=.o)

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

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

log.o : log.cpp log.hpp
parser.o: parser.cpp parser.hpp  
su_cdp.o : su_cdp.cpp su_cdp.hpp log.hpp
su_gather.o : su_gather.cpp su_gather.hpp log.hpp su_trace.hpp
su_trace.o : su_trace.cpp su_trace.hpp log.hpp
main.o : main.cu log.hpp utils.hpp parser.hpp su_gather.hpp
reference.o : reference.cpp utils.hpp

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -c0 1.98e-7 -c1 1.77e-6 -nc 5 -aph 600 -tau 0.002 -v 2 -i ./data/simple-synthetic.su
