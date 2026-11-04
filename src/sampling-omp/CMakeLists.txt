# sampling-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
