# concurrentKernels-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME concurrentKernels
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
