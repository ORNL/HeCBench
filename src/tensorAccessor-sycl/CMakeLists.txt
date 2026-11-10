# tensorAccessor-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorAccessor
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
