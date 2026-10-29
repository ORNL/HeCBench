# twell-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME twell
    MODEL sycl
    SOURCES main.cpp
    LINK_LIBRARIES -ldnnl
    CATEGORIES ml
    TEST_ARGS --reps 100
)
