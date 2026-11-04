# ccl-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ccl
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
    LINK_LIBRARIES ccl
)
