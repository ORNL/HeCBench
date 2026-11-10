# halo-finder-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME halo-finder
    MODEL cuda
    SOURCES ForceTreeTest.cxx
            ParticleDistribute.cxx
            ParticleExchange.cxx
            InitialExchange.cxx            
            Message.cxx
            ForceLaw.cxx
            BHForceTree.cxx
            RCOForceTree.cxx
            RCBForceTree.cxx
            Partition.cxx
            Timings.cxx
            Timer.cxx
            bigchunk.c
            BGQStep16.c
            #BGQCM.c
    CATEGORIES simulation
    INCLUDE_DIRS dfft
    COMPILE_OPTIONS -DRCB_UNTHREADED_BUILD -DUSE_SERIAL_COSMO -DID_64 -DPOSVEL_32 -DGRID_32 -DLONG_INTEGER
)
set_source_files_properties(
            ForceTreeTest.cxx
            ParticleDistribute.cxx
            ParticleExchange.cxx
            InitialExchange.cxx            
            Message.cxx
            ForceLaw.cxx
            BHForceTree.cxx
            RCOForceTree.cxx
            RCBForceTree.cxx
            Partition.cxx
            Timings.cxx
            Timer.cxx
            PROPERTIES LANGUAGE CUDA)
