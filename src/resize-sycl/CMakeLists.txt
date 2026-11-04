# resize-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME resize
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
