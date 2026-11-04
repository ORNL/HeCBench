# hausdorff-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hausdorff
    MODEL omp
    SOURCES main.cpp
    CATEGORIES geoscience
)
