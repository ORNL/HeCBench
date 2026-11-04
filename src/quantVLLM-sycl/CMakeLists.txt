# quantVLLM-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantVLLM
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
