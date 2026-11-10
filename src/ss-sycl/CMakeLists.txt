# ss-sycl/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/input.tar.gz")

add_hecbench_benchmark(
    NAME ss
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
)
