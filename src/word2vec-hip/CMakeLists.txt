# word2vec-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME word2vec
    MODEL hip
    SOURCES cbow.cu word2vec.cu
    CATEGORIES ml
    LINK_LIBRARIES pthread
)
