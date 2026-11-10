# msm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME msm
    MODEL sycl
    SOURCES main.cpp
    INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/../msm-cuda
    CATEGORIES cryptography
    TEST_REGEX "PASS"
    TEST_ARGS 1 12
)
