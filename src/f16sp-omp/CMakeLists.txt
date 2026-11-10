# f16sp-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16sp
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
