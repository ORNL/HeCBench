# bicgstab-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bicgstab
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
    LINK_LIBRARIES cublas cusparse
)
