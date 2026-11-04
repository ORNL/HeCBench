# chemv-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME chemv
    MODEL omp
    SOURCES main.cpp
    CATEGORIES simulation
)
