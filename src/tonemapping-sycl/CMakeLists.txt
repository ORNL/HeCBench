# tonemapping-sycl/CMakeLists.txt

REQUIREDATAFILE("${CMAKE_CURRENT_LIST_DIR}/../data/tonemapping/input.hdr")

add_hecbench_benchmark(
    NAME tonemapping
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
