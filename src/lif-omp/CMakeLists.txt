# lif-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lif
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
