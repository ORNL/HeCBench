# depixel-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME depixel
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
