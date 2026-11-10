# bsw-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bsw
    MODEL sycl
    SOURCES driver.cpp main.cpp utils.cpp
    CATEGORIES bioinformatics
)
