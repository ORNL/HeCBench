# qtclustering-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME qtclustering
    MODEL cuda
    SOURCES Option.cpp OptionParser.cpp QTC.cu comm.cpp libdata.cpp main.cpp
    CATEGORIES ml
)
