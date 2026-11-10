# atomicPerf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicPerf
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
