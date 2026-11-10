# triad-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME triad
    MODEL hip
    SOURCES Option.cpp OptionParser.cpp Timer.cpp main.cpp triad.cu
    CATEGORIES bandwidth
)
