# convolutionSeparable-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolutionSeparable
    MODEL sycl
    SOURCES conv.cpp conv_gold.cpp main.cpp
    CATEGORIES imaging
)
