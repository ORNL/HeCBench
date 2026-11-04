# reaction-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME reaction
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
