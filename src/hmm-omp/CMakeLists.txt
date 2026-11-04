# hmm-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME hmm
    MODEL omp
    SOURCES HiddenMarkovModel.cpp ViterbiCPU.cpp ViterbiGPU.cpp
    CATEGORIES math
)
