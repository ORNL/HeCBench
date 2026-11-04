# clenergy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME clenergy
    MODEL cuda
    SOURCES WKFUtils.cu clenergy.cu
    CATEGORIES simulation
)
