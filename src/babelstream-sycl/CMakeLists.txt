# babelstream-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME babelstream
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bandwidth
)
