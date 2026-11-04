# aobench-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME aobench
    MODEL omp
    SOURCES ao.cpp
    ENABLE_FASTMATH true
    CATEGORIES imaging
)
