# fresnel-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fresnel
    MODEL cuda
    SOURCES cosine.cu fresnel.cu main.cu sine.cu xchebyshev.cu
    CATEGORIES math
    COMPILE_OPTIONS -use_fast_math
)
