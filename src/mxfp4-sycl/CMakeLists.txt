# mxfp4-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mxfp4
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
