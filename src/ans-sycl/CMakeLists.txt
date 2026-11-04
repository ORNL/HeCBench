# ans-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ans
    MODEL sycl
    SOURCES
        src/ans_encoder.cc
        src/ans_table_generator.cc
        src/main.cc
        src/multians_codetable.cc
        src/multians_gpu_decoder.cc
        src/multians_input_buffer.cc
        src/multians_output_buffer.cc
        src/multians_util.cc
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include
    CATEGORIES encoding
)
