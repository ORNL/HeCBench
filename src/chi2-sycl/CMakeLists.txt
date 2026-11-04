# chi2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME chi2
    MODEL sycl
    SOURCES chi2.cpp
    CATEGORIES simulation
)
