# minimap2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minimap2
    MODEL cuda
    SOURCES
        src/host_data_io.cpp
        src/host_kernel.cpp
        src/main.cpp
        src/memory_scheduler.cpp
        device/device_kernel.cu
        device/device_kernel_wrapper.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    CATEGORIES bioinformatics
)
