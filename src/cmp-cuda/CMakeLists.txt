# cmp-cuda/CMakeLists.txt

UNZIPFILE("${CMAKE_CURRENT_LIST_DIR}/data.tar.bz")

add_hecbench_benchmark(
    NAME cmp
    MODEL cuda
    SOURCES log.cpp main.cu parser.cpp reference.cpp su_cdp.cpp su_gather.cpp su_trace.cpp
    CATEGORIES simulation
)
