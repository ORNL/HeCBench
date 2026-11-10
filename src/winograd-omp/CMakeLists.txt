# winograd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME winograd
    MODEL omp
    SOURCES main.cpp utils.cpp
    CATEGORIES ml
)
