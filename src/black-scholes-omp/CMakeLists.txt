# black-scholes-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME black-scholes
    MODEL omp
    SOURCES blackScholesAnalyticEngine.cpp
    CATEGORIES finance
)
