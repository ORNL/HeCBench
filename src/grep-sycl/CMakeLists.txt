# grep-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME grep
    MODEL sycl
    SOURCES main.cpp nfautil.cpp regex.cpp
    CATEGORIES search
)
