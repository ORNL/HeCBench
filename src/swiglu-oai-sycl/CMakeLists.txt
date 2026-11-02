# swiglu-oai-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME swiglu-oai
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES algorithms
)
