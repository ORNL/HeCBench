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

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../sad-cuda

# Linker Flags
LDFLAGS = 

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  LDFLAGS += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

# SAD images and bitmap_image.hpp come from DVC: src/sad-cuda/data.tar.gz
DATA_DIR := ../sad-cuda
DATA_TGZ := $(DATA_DIR)/data.tar.gz
DATA_HDR := $(DATA_DIR)/bitmap_image.hpp

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu $(DATA_HDR) Makefile
	$(CC) $(CFLAGS) -c $< -o $@

$(DATA_HDR): $(wildcard $(DATA_TGZ))
	@test -f $(DATA_TGZ) || { \
	  echo "Missing $(DATA_TGZ). From the repository root run: dvc pull src/sad-cuda/data.tar.gz" >&2; \
	  exit 1; }
	tar -xzf $(DATA_TGZ) -C $(DATA_DIR)
	touch $(DATA_HDR)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) ../sad-cuda/coins.bmp ../sad-cuda/coin.bmp 100
