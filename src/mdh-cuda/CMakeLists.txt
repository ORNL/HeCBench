# mdh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mdh
    MODEL cuda
    SOURCES WKFUtils.cu main.cu
    CATEGORIES simulation
)
