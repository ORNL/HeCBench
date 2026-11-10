# relu-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME relu
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
