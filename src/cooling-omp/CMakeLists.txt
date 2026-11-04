# cooling-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME cooling
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES simulation
)
