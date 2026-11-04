# feynman-kac-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME feynman-kac
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
