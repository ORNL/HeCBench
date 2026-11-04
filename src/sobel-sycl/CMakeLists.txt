# sobel-sycl/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME sobel
    MODEL sycl
    SOURCES main.cpp reference.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${SDK_DIR}
)
