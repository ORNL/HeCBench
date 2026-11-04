# rotary-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME rotary
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
