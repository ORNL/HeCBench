# dense-embedding-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dense-embedding
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
