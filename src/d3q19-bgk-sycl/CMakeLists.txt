# d3q19-bgk-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME d3q19-bgk
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
)
