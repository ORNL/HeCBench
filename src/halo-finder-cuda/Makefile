NVCC_BIN=nvcc

ARCH    = sm_60

# the path may be adjusted for each platform
MPI_INCLUDE ?= /usr/lib/x86_64-linux-gnu/openmpi/include
#MPI_INCLUDE=/usr/include/openmpi-x86_64/

OPT=-O3 -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO

#NVCC_FLAGS=-use_fast_math -DINLINE_FORCE -I${MPI_INCLUDE}
NVCC_FLAGS=-arch=$(ARCH) -I${MPI_INCLUDE}

HACC_PLATFORM=cuda
HACC_OBJDIR=${HACC_PLATFORM}

HACC_CFLAGS=${OPT} ${OMP} ${NVCC_FLAGS}
HACC_CC=nvcc -x c

HACC_CXXFLAGS=${OPT} ${OMP} ${NVCC_FLAGS}
HACC_CXX=nvcc

HACC_LDFLAGS=-lm -lrt

HACC_NUM_CUDA_DEV="1"
HACC_MPI_CFLAGS=${OPT} ${OMP} ${NVCC_FLAGS}
HACC_MPI_CC=nvcc -x c

HACC_MPI_CXXFLAGS=${OPT} ${OMP} ${NVCC_FLAGS}
HACC_MPI_CXX=nvcc
HACC_MPI_LD=nvcc

HACC_MPI_LDFLAGS=-lm -lrt

OBJDIR = ${HACC_OBJDIR}
#SOBJDIR = ${HACC_OBJDIR}_serial


PROGS    = \
	$(OBJDIR)/ForceTreeTest


LIBS	= \
	$(OBJDIR)/libpartition.a \
	$(OBJDIR)/libparticle.a \
	$(OBJDIR)/libBHForceTree.a \
	$(OBJDIR)/libhalotime.a \
	$(OBJDIR)/libbigchunk.a


# Probably need to fix dependencies on some of the header files.
HDR= \
	RCBForceTree.h \
	RCOForceTree.h \
	BHForceTree.h \
	CosmoHalo.h \
	Definition.h \
	ForceLaw.h \
	ForceTree.h \
	InitialExchange.h \
	Message.h \
	Partition.h \
	ParticleDistribute.h \
	ParticleExchange.h \
	Timer.h \
	Timings.h \
	cudaUtil.h


all: ${PROGS}
libs: ${LIBS}
progs: ${PROGS}

include include.mk
include dfft/include.mk
CXXFLAGS += ${DFFT_CXXFLAGS}

CFLAGS += ${HACC_MPI_CFLAGS}
CXXFLAGS += ${HACC_MPI_CXXFLAGS}
LDFLAGS += ${HACC_MPI_LDFLAGS}

#CXXFLAGS += ${MPI_COMPILE_FLAGS} 
CXXFLAGS += ${HF_TYPE_FLAGS}
CXXFLAGS += ${HF_WARNING}
#CXXFLAGS += -g
#CXXFLAGS += -DDEBUG
CXXFLAGS += -I.
CXXFLAGS += -x cu
#LDFLAGS  = ${MPI_LD_FLAGS} -lmpi_cxx

CFLAGS += -Idfft
CXXFLAGS += -Idfft


$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJDIR)/%.o: %.c | $(OBJDIR)
	${HACC_MPI_CC} ${CFLAGS} -c -o $@ $<

$(OBJDIR)/%.o: %.cxx | $(OBJDIR)
	${HACC_MPI_CXX} ${CXXFLAGS} -c -o $@ $<

$(OBJDIR)/%.o: %.cu | $(OBJDIR)
	${HACC_MPI_CXX} ${CFLAGS} -c -o $@ $<


#$(SOBJDIR):
#	mkdir -p $(SOBJDIR)
#
#$(SOBJDIR)/%.o: %.c | $(SOBJDIR)
#	${HACC_CC} -c -o $@ $<
#
#$(SOBJDIR)/%.o: %.cxx | $(SOBJDIR)
#	${HACC_CXX} -c -o $@ $<


#$(OBJDIR)/ForceLaw.o: ForceLaw.h


dims-local.c: dfft/dims.c
	cp -f dfft/dims.c dims-local.c

# Explicit prerequisite: dims-local.o needs dims-local.c to be generated first.
# Without this, 'make -j' can attempt to compile the object before 'cp' runs.
$(OBJDIR)/dims-local.o: dims-local.c

$(OBJDIR)/ForceTreeTest: $(OBJDIR)/ForceTreeTest.o $(OBJDIR)/libparticle.a $(OBJDIR)/libBHForceTree.a $(OBJDIR)/libpartition.a $(OBJDIR)/libhalotime.a $(OBJDIR)/libbigchunk.a $(OBJDIR)/dims-local.o
	${HACC_MPI_CXX} -o $@ $^ ${LDFLAGS} -DUSE_SERIAL_COSMO=1

# Use explicit object lists + ar/ranlib instead of the deprecated
# archive(member.o) syntax. The latter triggers Make's built-in .c.o and .o.a
# implicit rules (which call cc), bypassing this project's hipcc recipe and
# mis-compiling the objects.
$(OBJDIR)/libpartition.a: $(OBJDIR)/Partition.o
	ar rv $@ $(OBJDIR)/Partition.o
	ranlib $@


PARTICLE_SOURCES += ParticleDistribute.cxx
PARTICLE_SOURCES += ParticleExchange.cxx
PARTICLE_SOURCES += InitialExchange.cxx
PARTICLE_SOURCES += Message.cxx
PARTICLE_OBJLIST = $(PARTICLE_SOURCES:.cxx=.o)
PARTICLE_OBJECTS := $(addprefix $(OBJDIR)/,$(PARTICLE_OBJLIST))
$(OBJDIR)/libparticle.a: $(PARTICLE_OBJECTS)
	ar rv $@ $(PARTICLE_OBJECTS)
	ranlib $@


HT_SOURCES += Timings.cxx
HT_SOURCES += Timer.cxx
HT_OBJLIST = $(HT_SOURCES:.cxx=.o)
HT_OBJECTS := $(addprefix $(OBJDIR)/,$(HT_OBJLIST))
$(OBJDIR)/libhalotime.a: $(HT_OBJECTS)
	ar rv $@ $(HT_OBJECTS)
	ranlib $@


FORCE_SOURCES += ForceLaw.cxx
FORCE_SOURCES += BHForceTree.cxx
FORCE_SOURCES += RCOForceTree.cxx
FORCE_SOURCES += RCBForceTree.cxx
FORCE_SOURCES_C += BGQStep16.c
#FORCE_SOURCES_C += BGQCM.c
FORCE_OBJLIST = $(FORCE_SOURCES:.cxx=.o)
FORCE_OBJLIST_C = $(FORCE_SOURCES_C:.c=.o)
FORCE_OBJECTS := $(addprefix $(OBJDIR)/,$(FORCE_OBJLIST) $(FORCE_OBJLIST_C))
# Note: $(FORCE_OBJLIST_C) is already a subset of $(FORCE_OBJECTS) (added via
# addprefix above) — listing it again as a separate prerequisite causes Make
# to try its implicit cc rule for those objects.
$(OBJDIR)/libBHForceTree.a: $(FORCE_OBJECTS)
	ar rv $@ $(FORCE_OBJECTS)
	ranlib $@

$(OBJDIR)/libbigchunk.a: $(OBJDIR)/bigchunk.o
	ar rv $@ $(OBJDIR)/bigchunk.o
	ranlib $@

clean:
	rm -rf cuda
	rm -f dims-local.c
	rm -f *.o *.a ForceTreeTest 2>/dev/null || true

run: progs
	$(LAUNCHER) ./cuda/ForceTreeTest 0.5 0.1 10000 0.1 10 N 12 rcb
