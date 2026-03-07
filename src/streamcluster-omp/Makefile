CC        = icpx
OPTIMIZE  = yes
DEBUG     = no
DEVICE    = gpu
PROFILE   = yes
LAUNCHER  =


# Standard flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall

# Linker flags
LDFLAGS = 

ifeq ($(PROFILE),yes)
  CFLAGS += -DPROFILE_TMP
endif

# Enable checksum and output file
ifeq ($(DEBUG),yes)
  CFLAGS += -g 
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
endif

ifeq ($(DEVICE),gpu)
  CFLAGS +=-fiopenmp -fopenmp-targets=spir64 -D__STRICT_ANSI__
else
  CFLAGS +=-qopenmp
endif

SRC = streamcluster.cpp
EXE = main

$(EXE):$(SRC) kernel.h streamcluster.h  streamcluster_cl.h
	$(CC) $(CFLAGS) $(SRC) -o $(EXE) $(LDFLAGS)

run: $(EXE)
	$(LAUNCHER) ./$(EXE) 10 20 256 65536 65536 1000 none output.txt 1 

clean: 
	rm -f $(EXE) output.txt result*
