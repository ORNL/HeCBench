# fresnel-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME fresnel
    MODEL omp
    SOURCES cosine.cpp fresnel.cpp main.cpp sine.cpp xchebyshev.cpp
    CATEGORIES math
)
