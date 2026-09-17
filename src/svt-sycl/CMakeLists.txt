# svt-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME svt
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES simulation
    TEST_ARGS 12 1280 720 100
    TEST_REGEX "Raycast: PASS"
    TEST_TIMEOUT 300
)
