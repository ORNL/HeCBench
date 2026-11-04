# simpleMultiDevice-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleMultiDevice
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
