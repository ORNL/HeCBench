# depixel-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME depixel
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
