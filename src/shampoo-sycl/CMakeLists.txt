# shampoo-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME shampoo
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
    COMPILE_OPTIONS "SHELL:-Xarch_host -fopenmp"
    LINK_LIBRARIES -qmkl=parallel -Xarch_host -fopenmp
    TEST_ARGS 33 31 3 1
    TEST_REGEX "PASS"
)
