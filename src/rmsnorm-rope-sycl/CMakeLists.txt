# rmsnorm-rope-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm-rope
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
