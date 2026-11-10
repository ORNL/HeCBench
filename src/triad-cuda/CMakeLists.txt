# triad-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME triad
    MODEL cuda
    SOURCES Option.cpp OptionParser.cpp Timer.cpp main.cpp triad.cu
    CATEGORIES bandwidth
)
