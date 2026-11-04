# resize-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME resize
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
