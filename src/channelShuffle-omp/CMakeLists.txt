# channelShuffle-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME channelShuffle
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
