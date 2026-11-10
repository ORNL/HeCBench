# boxfilter-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME boxfilter
    MODEL sycl
    SOURCES cmd_arg_reader.cpp main.cpp reference.cpp shrUtils.cpp
    CATEGORIES imaging
)
