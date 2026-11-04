# mixbench-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mixbench
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
