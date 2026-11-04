# mask-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME mask
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
