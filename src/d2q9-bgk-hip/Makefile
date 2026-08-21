#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=

CheckSize ?= 256x256
FINAL_STATE_FILE = ./final_state.dat
AV_VELS_FILE = ./av_vels.dat
REF_FINAL_STATE_FILE = ./check/$(CheckSize).final_state.dat
REF_AV_VELS_FILE = ./check/$(CheckSize).av_vels.dat


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cu

obj = $(source:.cu=.o)

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

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) *.dat

run: $(program)
	$(LAUNCHER) ./$(program) Inputs/input_256x256.params Obstacles/obstacles_256x256.dat

check: $(AV_VELS_FILE) $(FINAL_STATE_FILE)
	python check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) \
        --ref-final-state-file=$(REF_FINAL_STATE_FILE) \
        --av-vels-file=$(AV_VELS_FILE) \
        --final-state-file=$(FINAL_STATE_FILE)
