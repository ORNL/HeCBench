# stencil1d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil1d
    MODEL omp
    SOURCES stencil_1d.cpp
    CATEGORIES imaging
)
