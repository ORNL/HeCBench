# putget-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME putget
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
