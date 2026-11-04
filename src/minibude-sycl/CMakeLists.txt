# minibude-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.gz")

add_hecbench_benchmark(
    NAME minibude
    MODEL sycl
    SOURCES kernel.cpp main.cpp
    CATEGORIES bioinformatics
)
