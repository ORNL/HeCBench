# mallocFree-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mallocFree
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
