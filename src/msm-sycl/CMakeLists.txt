# msm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME msm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography
)
