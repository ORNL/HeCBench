# threadfence-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME threadfence
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
