# geam-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME geam
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
