# page-rank-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME page-rank
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
