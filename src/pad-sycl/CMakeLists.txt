# pad-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME pad
    MODEL sycl
    SOURCES host_task.cpp kernel.cpp main.cpp
    CATEGORIES language
    LINK_LIBRARIES pthread
)
