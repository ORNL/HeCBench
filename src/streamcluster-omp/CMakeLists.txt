# streamcluster-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamcluster
    MODEL omp
    SOURCES streamcluster.cpp
    CATEGORIES ml
)
