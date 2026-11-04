# atomicPerf-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicPerf
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
