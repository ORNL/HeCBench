# bn-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
