# glu-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME glu
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
