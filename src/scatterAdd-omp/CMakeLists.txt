# scatterAdd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME scatterAdd
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
