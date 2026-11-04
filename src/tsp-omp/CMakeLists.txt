# tsp-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsp
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES search
)
