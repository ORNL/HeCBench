# nms-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nms
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
