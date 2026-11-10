# minimod-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME minimod
    MODEL sycl
    SOURCES constants.cpp data_setup.cpp grid.cpp main.cpp minimig.cpp pml.cpp
    CATEGORIES simulation
)
