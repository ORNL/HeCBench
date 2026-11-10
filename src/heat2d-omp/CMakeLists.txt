# heat2d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat2d
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
