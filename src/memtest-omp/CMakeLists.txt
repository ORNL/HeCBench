# memtest-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME memtest
    MODEL omp
    SOURCES main.cpp
    CATEGORIES bandwidth
)
