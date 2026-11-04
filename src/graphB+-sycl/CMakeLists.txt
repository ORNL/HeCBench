# graphB+-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME graphB+
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES graph
)
