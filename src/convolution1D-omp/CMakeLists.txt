# convolution1D-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution1D
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
