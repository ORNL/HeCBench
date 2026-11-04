# dispatch-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dispatch
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
