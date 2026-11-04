# layout-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME layout
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
