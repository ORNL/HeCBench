# gelu-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gelu
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
