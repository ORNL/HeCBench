# ced-sycl/CMakeLists.txt

set(SRC_DIR "${CMAKE_CURRENT_LIST_DIR}")
UNZIPFILE("${SRC_DIR}/frames.tar.gz")

add_hecbench_benchmark(
    NAME ced
    MODEL sycl
    SOURCES kernel_reference.cpp main.cpp
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
