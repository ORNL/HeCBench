# mallocFree-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mallocFree
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
