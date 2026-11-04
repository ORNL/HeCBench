# kernelLaunch-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME kernelLaunch
    MODEL omp
    SOURCES main.cpp
    CATEGORIES language
)
