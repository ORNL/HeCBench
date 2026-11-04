# lombscargle-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lombscargle
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES signal
)
