# nms-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nms
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
