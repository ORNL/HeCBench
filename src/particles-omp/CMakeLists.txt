# particles-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME particles
    MODEL omp
    SOURCES main.cpp bitonicSort.cpp particles.cpp
    CATEGORIES simulation
)
