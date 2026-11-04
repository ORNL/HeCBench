# channelSum-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelSum
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
