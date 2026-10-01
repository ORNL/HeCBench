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

source = main.cpp kernels.cu util.cpp parse.cpp sequential.cpp

obj = main.o kernels.o util.o parse.o sequential.o

#===============================================================================
# Sets Flags
#===============================================================================

# Standard Flags
CFLAGS := $(EXTRA_CFLAGS) -std=c++17 -Wall -I../hbc-cuda

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

main.o: ../hbc-cuda/main.cpp ../hbc-cuda/parse.h ../hbc-cuda/sequential.h \
        ../hbc-cuda/util.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

kernels.o: kernels.cu ../hbc-cuda/util.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

util.o: ../hbc-cuda/util.cpp ../hbc-cuda/util.h Makefile 
	$(CC) $(CFLAGS) -c $< -o $@

sequential.o: ../hbc-cuda/sequential.cpp ../hbc-cuda/sequential.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

parse.o: ../hbc-cuda/parse.cpp ../hbc-cuda/parse.h Makefile
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj) output.txt

run: $(program)
	$(LAUNCHER) ./$(program) -i ../hbc-cuda/graphs/breast_coexpress.txt -v --printscores=output.txt
