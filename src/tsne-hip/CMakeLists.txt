# tsne-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsne
    MODEL hip
    SOURCES apply_forces.cu attr_forces.cu cuda_utils.cu debug_utils.cu distance_utils.cu fit_tsne.cu main.cu math_utils.cu matrix_broadcast_utils.cu nbodyfft.cu perplexity_search.cu rep_forces.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES ml
)
