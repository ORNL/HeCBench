# expdist-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME expdist
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
