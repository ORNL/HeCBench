# vdi-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME vdi
    MODEL cuda
    SOURCES main.cu fused.cu libraries.cu
    CATEGORIES math
    LINK_LIBRARIES cublas cusolver
)
