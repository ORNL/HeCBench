# tensorT-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorT
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
