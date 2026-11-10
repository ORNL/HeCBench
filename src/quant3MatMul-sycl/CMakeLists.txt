# quant3MatMul-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME quant3MatMul
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
