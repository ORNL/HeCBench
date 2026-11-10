# black-scholes-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME black-scholes
    MODEL cuda
    SOURCES blackScholesAnalyticEngine.cu
    CATEGORIES finance
)
