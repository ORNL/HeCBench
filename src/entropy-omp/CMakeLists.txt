# entropy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME entropy
    MODEL omp
    SOURCES main.cpp
    CATEGORIES encoding
)
