#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = nvcc
OPTIMIZE  = yes
DEBUG     = no
ARCH      = sm_60
LAUNCHER  =


#===============================================================================
# Program name & source code list
#===============================================================================

program = main

cpp_source = main.cpp Option.cpp OptionParser.cpp comm.cpp libdata.cpp
cuda_source = QTC.cu 

obj = $(cpp_source:.cpp=.o) $(cuda_source:.cu=.o)

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Xcompiler -Wall -arch=$(ARCH)

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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

main.o: main.cpp OptionParser.h Utility.h 
	$(CC) $(CFLAGS) -c $< -o $@

comm.o: comm.cpp comm.h
	$(CC) $(CFLAGS) -c $< -o $@

libdata.o: libdata.cpp libdata.h
	$(CC) $(CFLAGS) -c $< -o $@

Option.o: Option.cpp Option.h 
	$(CC) $(CFLAGS) -c $< -o $@

OptionParser.o: OptionParser.cpp OptionParser.h Utility.h
	$(CC) $(CFLAGS) -c $< -o $@

QTC.o: QTC.cu cudacommon.h qtclib.h OptionParser.h libdata.h \
       tuningParameters.h comm.h kernels_compact_storage.h kernels_common.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) --Verbose
	#./$(program) --SaveOutput
