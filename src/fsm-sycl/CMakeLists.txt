# fsm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fsm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES bioinformatics
)
