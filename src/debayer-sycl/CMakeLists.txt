# debayer-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME debayer
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
