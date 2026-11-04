# henry-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME henry
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES simulation
)
