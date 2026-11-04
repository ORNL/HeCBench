# langford-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME langford
    MODEL omp
    SOURCES main.cpp
    CATEGORIES math
)
