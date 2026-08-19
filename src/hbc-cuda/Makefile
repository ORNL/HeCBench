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

source = main.cpp kernels.cu util.cpp parse.cpp sequential.cpp

obj = main.o kernels.o util.o parse.o sequential.o

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

main.o: main.cpp parse.h sequential.h util.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cu util.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

util.o: util.cpp util.h Makefile 
	$(CC) $(CFLAGS) -c $< -o $@

sequential.o: sequential.cpp sequential.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

parse.o: parse.cpp parse.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output.txt

run: $(program)
	$(LAUNCHER) ./$(program) -i ./graphs/breast_coexpress.txt -v --printscores=output.txt
