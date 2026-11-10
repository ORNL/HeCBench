# hmm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hmm
    MODEL sycl
    SOURCES HiddenMarkovModel.cpp ViterbiCPU.cpp ViterbiGPU.cpp
    CATEGORIES math
)
