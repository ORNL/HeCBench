# langevin-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME langevin
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES math
)
