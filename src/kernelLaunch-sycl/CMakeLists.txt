# kernelLaunch-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME kernelLaunch
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
