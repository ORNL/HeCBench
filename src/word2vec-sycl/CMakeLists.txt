# word2vec-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME word2vec
    MODEL sycl
    SOURCES cbow.cpp word2vec.cpp
    CATEGORIES ml
    LINK_LIBRARIES pthread
)
