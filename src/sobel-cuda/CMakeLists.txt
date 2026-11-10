# sobel-cuda/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

add_hecbench_benchmark(
    NAME sobel
    MODEL cuda
    SOURCES main.cu reference.cu
    CATEGORIES imaging
    INCLUDE_DIRS ${SDK_DIR}
)
