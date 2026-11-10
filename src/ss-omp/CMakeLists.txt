# ss-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME ss
    MODEL omp
    SOURCES main.cpp
    CATEGORIES search
)
