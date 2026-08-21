#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = g++
HIPCC     = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  ?=
ROCM_PATH ?= /opt/rocm

#===============================================================================
# Program name & source code list
#===============================================================================

program = gerbil

obj = AddKernel.o        \
      CompressKernel.o   \
      FastParser.o       \
      global.o           \
      KmerDistributor.o  \
      SequenceSplitter.o \
      TempFile.o         \
      Application.o      \
      debug.o            \
      FastReader.o       \
      SuperReader.o      \
      Bundle.o           \
      FastFile.o         \
      gerbil.o           \
      KmcWriter.o        \
      SuperWriter.o      

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -fpermissive -DGERBIL_USE_GPU \
	      -DBOOST_ALL_NO_LIB -DBOOST_ATOMIC_DYN_LINK -DBOOST_FILESYSTEM_DYN_LINK \
          -DBOOST_REGEX_DYN_LINK -DBOOST_SYSTEM_DYN_LINK -DBOOST_THREAD_DYN_LINK \
          -I$(ROCM_PATH)/include -D__HIP_PLATFORM_AMD__

HIPCCFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -DGERBIL_USE_GPU

# Linker Flags
LDFLAGS = -lboost_system -lboost_thread -lboost_regex -lboost_filesystem \
          -lbz2 -lz -L$(ROCM_PATH)/lib -lamdhip64

# Debug Flags
ifeq ($(DEBUG),yes)
  CFLAGS += -g -DDEBUG
  HIPCCFLAGS += -g -DDEBUG
  LDFLAGS  += -g
endif

# Optimization Flags
ifeq ($(OPTIMIZE),yes)
  CFLAGS += -O3
  HIPCCFLAGS += -O3
endif
#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
	$(HIPCC) $(obj) -o $@ $(LDFLAGS)

AddKernel.o:        src/cuda_ds/AddKernel.cu
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@
CompressKernel.o:   src/cuda_ds/CompressKernel.cu  
	$(HIPCC) $(HIPCCFLAGS) -c $< -o $@
FastParser.o:       src/gerbil/FastParser.cpp
	$(CC) $(CFLAGS) -c $< -o $@
global.o:           src/gerbil/global.cpp
	$(CC) $(CFLAGS) -c $< -o $@
KmerDistributor.o:  src/gerbil/KmerDistributor.cpp
	$(CC) $(CFLAGS) -c $< -o $@
SequenceSplitter.o: src/gerbil/SequenceSplitter.cpp
	$(CC) $(CFLAGS) -c $< -o $@
TempFile.o:         src/gerbil/TempFile.cpp
	$(CC) $(CFLAGS) -c $< -o $@
Application.o:      src/gerbil/Application.cpp
	$(CC) $(CFLAGS) -c $< -o $@
debug.o:            src/gerbil/debug.cpp
	$(CC) $(CFLAGS) -c $< -o $@
FastReader.o:       src/gerbil/FastReader.cpp
	$(CC) $(CFLAGS) -c $< -o $@
SuperReader.o:      src/gerbil/SuperReader.cpp
	$(CC) $(CFLAGS) -c $< -o $@
Bundle.o:           src/gerbil/Bundle.cpp
	$(CC) $(CFLAGS) -c $< -o $@
FastFile.o:         src/gerbil/FastFile.cpp
	$(CC) $(CFLAGS) -c $< -o $@
gerbil.o:           src/gerbil/gerbil.cpp
	$(CC) $(CFLAGS) -c $< -o $@
KmcWriter.o:        src/gerbil/KmcWriter.cpp
	$(CC) $(CFLAGS) -c $< -o $@
SuperWriter.o:      src/gerbil/SuperWriter.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) out

run: $(program)
	$(LAUNCHER) HIP_VISIBLE_DEVICES=0 ./$(program) -i -g ERR164407.fastq.gz tmp out
