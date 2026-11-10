# intrinsics-cast-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME intrinsics-cast
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
