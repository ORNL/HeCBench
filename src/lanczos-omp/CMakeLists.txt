# lanczos-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lanczos
    MODEL omp
    SOURCES lanczos.cpp main.cpp
    CATEGORIES math
)
