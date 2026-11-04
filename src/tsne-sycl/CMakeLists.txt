# tsne-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsne
    MODEL sycl
    SOURCES apply_forces.cpp attr_forces.cpp cuda_utils.cpp debug_utils.cpp distance_utils.cpp fit_tsne.cpp main.cpp math_utils.cpp matrix_broadcast_utils.cpp nbodyfft.cpp perplexity_search.cpp rep_forces.cpp
    CATEGORIES ml
    COMPILE_OPTIONS -ffast-math
)
