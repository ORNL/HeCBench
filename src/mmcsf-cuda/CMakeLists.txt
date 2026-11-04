# mmcsf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mmcsf
    MODEL cuda
    SOURCES kernels.cu main.cu mttkrp_cpu.cu
    CATEGORIES ml
    INCLUDE_DIRS ${Boost_INCLUDE_DIRS}
    LINK_LIBRARIES pthread
)
