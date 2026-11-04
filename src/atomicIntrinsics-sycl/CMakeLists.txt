# atomicIntrinsics-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicIntrinsics
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
