# background-subtract-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME background-subtract
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
