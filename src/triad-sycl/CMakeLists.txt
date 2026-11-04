# triad-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME triad
    MODEL sycl
    SOURCES Option.cpp OptionParser.cpp Timer.cpp main.cpp triad.cpp
    CATEGORIES bandwidth
)
