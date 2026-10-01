# mlaDecode-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlaDecode
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
