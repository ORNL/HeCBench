# base64e-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME base64e
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES encoding
)
