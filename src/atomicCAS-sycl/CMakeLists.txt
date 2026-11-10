# atomicCAS-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCAS
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
