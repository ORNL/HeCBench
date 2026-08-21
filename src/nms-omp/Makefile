#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC          = icpx
OPTIMIZE    = yes
DEBUG       = no
DEVICE      = gpu
BYTE_ATOMIC = no
LAUNCHER    ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp

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
  CFLAGS += -g
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(BYTE_ATOMIC),yes)
  CFLAGS += -DBYTE_ATOMIC
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__ 
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output.txt

run: $(program)
	$(LAUNCHER) ./$(program) ../nms-cuda/detections.txt output.txt 1000
