# ced-sycl/CMakeLists.txt

REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/ced/input")
REQUIREDATADIR("${CMAKE_CURRENT_LIST_DIR}/../data/ced/output")

add_hecbench_benchmark(
    NAME ced
    MODEL sycl
    SOURCES kernel_reference.cpp main.cpp
    CATEGORIES imaging
    LINK_LIBRARIES pthread
)
