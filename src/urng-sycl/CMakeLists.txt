# urng-sycl/CMakeLists.txt

set(SDK_DIR "${CMAKE_CURRENT_LIST_DIR}/../include")

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/image.tar.gz")

add_hecbench_benchmark(
    NAME urng
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES rng
    INCLUDE_DIRS ${SDK_DIR}
)
