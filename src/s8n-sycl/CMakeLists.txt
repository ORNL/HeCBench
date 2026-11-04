# s8n-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME s8n
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES search
)
