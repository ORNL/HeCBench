# idivide-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME idivide
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
