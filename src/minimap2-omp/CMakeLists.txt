# minimap2-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME minimap2
    MODEL omp
    SOURCES
        device/device_kernel_wrapper.cpp
        src/host_data_io.cpp
        src/host_kernel.cpp
        src/main.cpp
        src/memory_scheduler.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    CATEGORIES bioinformatics algorithms
)
