# floydwarshall-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME floydwarshall
    MODEL omp
    SOURCES main.cpp
    CATEGORIES graph
)
