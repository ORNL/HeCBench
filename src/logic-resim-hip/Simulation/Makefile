#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = src/gate/*.cpp src/main/*.cpp src/parser/*.cpp \
         src/sim/*.cu src/util/*.cpp src/wave/*.cpp

header = src/gate/*.h src/parser/*.h \
         src/sim/*.h src/util/*.h src/wave/*.h

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall \
          -Isrc/gate -Isrc/parser -Isrc/sim -Isrc/util -Isrc/wave

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

$(program): $(source) $(header) Makefile
	$(CC) $(CFLAGS) $(source) -o $@ $(LDFLAGS)

clean:
	rm -rf $(program) *.saif

run: $(program)
	$(LAUNCHER) ./$(program) ../../logic-resim-cuda/Simulation/toy/intermediate.file \
	../../logic-resim-cuda/Simulation/toy/NV_NVDLA_partition_o_dc_24x33x55_5x5x55x25_int8_input.vcd \
	0 2972036001 \
	toy-result.saif
