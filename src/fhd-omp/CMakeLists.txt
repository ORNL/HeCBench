# fhd-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fhd
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
