# mask-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mask
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
