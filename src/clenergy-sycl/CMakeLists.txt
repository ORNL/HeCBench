# clenergy-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME clenergy
    MODEL sycl
    SOURCES WKFUtils.cpp clenergy.cpp
    CATEGORIES simulation
)
