# zoom-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME zoom
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
