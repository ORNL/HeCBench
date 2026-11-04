# complex-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME complex
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
