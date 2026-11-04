# d2q9-bgk-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME d2q9-bgk
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
