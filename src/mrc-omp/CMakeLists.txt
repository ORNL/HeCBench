# mrc-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrc
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
