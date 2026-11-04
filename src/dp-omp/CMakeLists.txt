# dp-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp
    MODEL omp
    SOURCES main.cpp shrUtils.cpp
    CATEGORIES math
)
