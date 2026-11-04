# lrn-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME lrn
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
