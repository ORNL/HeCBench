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

source = cOgrd.cpp  cOkadaEarthquake.cpp  cOkadaFault.cpp  cSphere.cpp  main.cpp  okada.cpp  utilits.cpp

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

ifeq ($(DEVICE),gpu)
  CFLAGS += -fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $< -o $@

cOgrd.o: cOgrd.cpp  cOgrd.h utilits.h
cOkadaEarthquake.o: cOkadaEarthquake.cpp  cOkadaEarthquake.h utilits.h
cOkadaFault.o: cOkadaFault.cpp  cOkadaFault.h utilits.h
cSphere.o: cSphere.cpp cSphere.h utilits.h
main.o : main.cpp easywave.h utilits.h
utilits.o: utilits.cpp utilits.h

clean:
	rm -rf $(program) $(obj) eWave.2D.*

run: $(program)
	$(LAUNCHER) ./$(program) -grid ./data/grids/e2Asean.grd -source ./data/faults/BengkuluSept2007.flt -time 120 

