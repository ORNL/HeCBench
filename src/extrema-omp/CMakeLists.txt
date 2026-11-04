# extrema-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME extrema
    MODEL omp
    SOURCES main.cpp
    CATEGORIES signal
)
