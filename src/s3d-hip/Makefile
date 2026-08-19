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

source = main.cu  Option.cu  OptionParser.cu  S3D.cu

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

$(program): $(obj) Makefile
	$(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)


main.o: main.cu OptionParser.h Utility.h 
	$(CC) $(CFLAGS) -c $< -o $@

Option.o: Option.cu Option.h 
	$(CC) $(CFLAGS) -c $< -o $@

OptionParser.o: OptionParser.cu OptionParser.h  Utility.h
	$(CC) $(CFLAGS) -c $< -o $@

S3D.o: S3D.cu S3D.h hipcommon.h OptionParser.h gr_base.h ratt.h \
       ratt2.h ratx.h qssa.h qssa2.h rdwdot.h 
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(program) $(obj)

run: $(program)
	$(LAUNCHER) ./$(program) -q -n 100 -s 1

