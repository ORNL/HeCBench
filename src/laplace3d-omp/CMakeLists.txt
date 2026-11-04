# laplace3d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace3d
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
