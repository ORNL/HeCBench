# hmm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hmm
    MODEL hip
    SOURCES HiddenMarkovModel.cu ViterbiCPU.cu ViterbiGPU.cu
    CATEGORIES math
)
