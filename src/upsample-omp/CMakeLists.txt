# upsample-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME upsample
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
