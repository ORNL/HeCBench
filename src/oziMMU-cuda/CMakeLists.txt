# oziMMU-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME oziMMU
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas
)
