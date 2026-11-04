# f16atomic-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16atomic
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
