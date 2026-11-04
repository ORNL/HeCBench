# zoom-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME zoom
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
