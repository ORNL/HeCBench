# minmax-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minmax
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
