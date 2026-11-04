# minimap2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME minimap2
    MODEL hip
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
