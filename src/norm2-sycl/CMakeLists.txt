# norm2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME norm2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
    LINK_LIBRARIES -qmkl=parallel
)
