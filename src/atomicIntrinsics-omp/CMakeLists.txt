# atomicIntrinsics-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicIntrinsics
    MODEL omp
    SOURCES main.cpp
    CATEGORIES compression
)
