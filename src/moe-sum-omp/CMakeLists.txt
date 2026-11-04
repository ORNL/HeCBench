# moe-sum-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-sum
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
