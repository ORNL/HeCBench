# memcpy-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME memcpy
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bandwidth
)
