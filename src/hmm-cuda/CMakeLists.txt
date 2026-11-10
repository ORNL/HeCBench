# hmm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hmm
    MODEL cuda
    SOURCES HiddenMarkovModel.cu ViterbiCPU.cu ViterbiGPU.cu
    CATEGORIES math
)
