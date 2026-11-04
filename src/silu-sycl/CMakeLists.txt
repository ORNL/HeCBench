# silu-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME silu
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
