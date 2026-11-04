# atomicCost-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCost
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
