# mlp-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mlp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    LINK_LIBRARIES -ldnnl
)
