# filter-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME filter
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
