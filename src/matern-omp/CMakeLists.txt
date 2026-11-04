# matern-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME matern
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
