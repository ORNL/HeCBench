# maxpool3d-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxpool3d
    MODEL omp
    SOURCES main.cpp
    CATEGORIES ml
)
