# fpdc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpdc
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
