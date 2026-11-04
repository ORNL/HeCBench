# mixbench-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mixbench
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
