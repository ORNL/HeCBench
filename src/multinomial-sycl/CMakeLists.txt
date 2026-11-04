# multinomial-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME multinomial
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
