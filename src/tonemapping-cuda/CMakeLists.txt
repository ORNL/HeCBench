# tonemapping-cuda/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/tonemapping/input.hdr")

add_hecbench_benchmark(
    NAME tonemapping
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
