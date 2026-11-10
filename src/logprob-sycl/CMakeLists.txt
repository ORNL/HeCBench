# logprob-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME logprob
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS -ffast-math
)
