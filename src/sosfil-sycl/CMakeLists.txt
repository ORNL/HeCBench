# sosfil-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME sosfil
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES signal
)
