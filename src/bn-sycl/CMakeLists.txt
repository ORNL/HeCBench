# bn-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
