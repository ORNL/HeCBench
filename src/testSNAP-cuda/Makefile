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

source = main.cu

obj = $(source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -I. -I../testSNAP-omp \
          -DREFDATA_TWOJ=14  -arch=$(ARCH)

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

# SNAP reference headers are stored in testSNAP-omp (DVC: refdata.tar.gz)
REFDATA_DIR := ../testSNAP-omp
REFDATA_TGZ := $(REFDATA_DIR)/refdata.tar.gz
REFDATA_HDR := $(REFDATA_DIR)/refdata_2J2_W.h

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cu snap.h utils.cu $(REFDATA_HDR)
	$(CC) $(CFLAGS) -c $< -o $@

$(REFDATA_HDR): $(wildcard $(REFDATA_TGZ))
	@test -f $(REFDATA_TGZ) || { \
	  echo "Missing $(REFDATA_TGZ). From the repository root run: dvc pull src/testSNAP-omp/refdata.tar.gz" >&2; \
	  exit 1; }
	tar -xzf $(REFDATA_TGZ) -C $(REFDATA_DIR)

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --nsteps 100
