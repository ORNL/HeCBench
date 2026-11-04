# maxFlops-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxFlops
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
