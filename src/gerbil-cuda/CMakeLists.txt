# gerbil-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gerbil
    MODEL cuda
    SOURCES src/cuda_ds/AddKernel.cu
            src/cuda_ds/CompressKernel.cu  
            src/gerbil/FastParser.cpp
            src/gerbil/global.cpp
            src/gerbil/KmerDistributor.cpp
            src/gerbil/SequenceSplitter.cpp
            src/gerbil/TempFile.cpp
            src/gerbil/Application.cpp
            src/gerbil/debug.cpp
            src/gerbil/FastReader.cpp
            src/gerbil/SuperReader.cpp
            src/gerbil/Bundle.cpp
            src/gerbil/FastFile.cpp
            src/gerbil/gerbil.cpp
            src/gerbil/KmcWriter.cpp
            src/gerbil/SuperWriter.cpp
    CATEGORIES bioinformatics
    COMPILE_OPTIONS -DGPU -fpermissive
	            -DBOOST_ALL_NO_LIB -DBOOST_ATOMIC_DYN_LINK -DBOOST_FILESYSTEM_DYN_LINK
	            -DBOOST_REGEX_DYN_LINK -DBOOST_SYSTEM_DYN_LINK -DBOOST_THREAD_DYN_LINK
    LINK_LIBRARIES z bz2 boost_system boost_thread boost_regex boost_filesystem
)
