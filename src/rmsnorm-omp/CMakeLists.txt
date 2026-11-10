# rmsnorm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
