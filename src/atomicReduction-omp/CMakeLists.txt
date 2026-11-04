# atomicReduction-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicReduction
    MODEL omp
    SOURCES reduction.cpp
    CATEGORIES compression
)
