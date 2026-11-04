# vol2col-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME vol2col
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
