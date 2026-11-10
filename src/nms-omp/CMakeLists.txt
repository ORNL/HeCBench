# nms-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME nms
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
