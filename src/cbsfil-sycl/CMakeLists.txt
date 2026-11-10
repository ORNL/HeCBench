# cbsfil-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cbsfil
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES imaging
)
