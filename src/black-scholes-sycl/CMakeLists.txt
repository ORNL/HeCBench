# black-scholes-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME black-scholes
    MODEL sycl
    SOURCES blackScholesAnalyticEngine.cpp
    CATEGORIES finance
)
