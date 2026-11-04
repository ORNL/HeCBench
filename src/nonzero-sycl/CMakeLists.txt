# nonzero-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME nonzero
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
