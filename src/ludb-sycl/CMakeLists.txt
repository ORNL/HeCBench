# ludb-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ludb
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
