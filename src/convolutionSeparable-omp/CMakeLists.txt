# convolutionSeparable-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME convolutionSeparable
    MODEL omp
    SOURCES conv.cpp conv_gold.cpp main.cpp
    CATEGORIES imaging
)
