# base64e-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME base64e
    MODEL cuda
    SOURCES main.cu
    CATEGORIES encoding
)
