# black-scholes-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME black-scholes
    MODEL hip
    SOURCES blackScholesAnalyticEngine.cu
    CATEGORIES finance
)
