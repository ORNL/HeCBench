# tridiagonal-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tridiagonal
    MODEL sycl
    SOURCES cmd_arg_reader.cpp main.cpp shrUtils.cpp
    CATEGORIES simulation
    COMPILE_OPTIONS -DUNIX -DBENCH_ITERATIONS=100
)
