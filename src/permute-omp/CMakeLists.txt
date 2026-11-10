# permute-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME permute
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
