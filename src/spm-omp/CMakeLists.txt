# spm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME spm
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
