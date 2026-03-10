#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = hipcc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

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
# Program name & source code list
#===============================================================================
program = main multimat multimat_F

all: $(program)

source=compact.cu full_matrix.cu multimat.cu

multimat: $(source)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
multimat_F: $(source)
	$(CC) $(CFLAGS) -DFUSED $^ -o $@ $(LDFLAGS)
main: $(source)
	$(CC) $(CFLAGS) -DFUSED -DLINKED $^ -o $@ $(LDFLAGS)

clean:
	rm -rf $(program)

run: $(program)
	$(LAUNCHER) ./main 2000 2000
	$(LAUNCHER) ./multimat_F 2000 2000
	$(LAUNCHER) ./multimat 2000 2000
	$(LAUNCHER) ./main 2000 2000 0.3 0.05 0.05
	$(LAUNCHER) ./multimat_F 2000 2000 0.3 0.05 0.05
	$(LAUNCHER) ./multimat 2000 2000 0.3 0.05 0.05
