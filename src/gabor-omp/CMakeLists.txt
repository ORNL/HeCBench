# gabor-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME gabor
    MODEL omp
    SOURCES main.cpp
    ENABLE_FASTMATH true
    CATEGORIES imaging
)
