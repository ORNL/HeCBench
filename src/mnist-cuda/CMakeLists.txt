# mnist-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.zip")

add_hecbench_benchmark(
    NAME mnist
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
