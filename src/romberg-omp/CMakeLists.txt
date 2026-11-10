# romberg-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME romberg
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
