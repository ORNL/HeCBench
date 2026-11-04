# gmm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME gmm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
