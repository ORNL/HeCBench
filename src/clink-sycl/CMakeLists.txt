# clink-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME clink
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
