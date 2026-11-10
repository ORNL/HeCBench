# grrt-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME grrt
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES simulation
)
