# mmcsf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mmcsf
    MODEL sycl
    SOURCES kernels.cpp main.cpp mttkrp_cpu.cpp
    CATEGORIES ml
    INCLUDE_DIRS ${Boost_INCLUDE_DIRS}
    LINK_LIBRARIES pthread
)
