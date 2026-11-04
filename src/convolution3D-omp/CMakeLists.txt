# convolution3D-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolution3D
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
