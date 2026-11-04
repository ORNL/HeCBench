# graphExecution-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME graphExecution
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
