# snicit-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME snicit
    MODEL cuda
    SOURCES src/beyond.cu
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party
    COMPILE_OPTIONS --extended-lambda
    CATEGORIES ml
)
