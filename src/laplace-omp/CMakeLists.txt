# laplace-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
