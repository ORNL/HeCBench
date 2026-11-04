# stencil3d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME stencil3d
    MODEL omp
    SOURCES main.cpp
    CATEGORIES imaging
)
