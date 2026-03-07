CXX = g++
NVCC = nvcc
ARCH = sm_60

# path #
SRC_PATH = src
CUDA_PATH = device
BUILD_PATH = build
BUILD_CUDA_PATH = build/cuda
BIN_PATH = $(BUILD_PATH)/bin

# executable #
BIN_NAME = main

# extensions #
SRC_EXT = cpp
CUDA_EXT = cu

# code lists #
# Find all source files in the source directory, sorted by
# most recently modified
SOURCES = $(shell find $(SRC_PATH) -name '*.$(SRC_EXT)' | sort -k 1nr | cut -f2-)
CUDA_SOURCES = $(shell find $(CUDA_PATH) -name '*.$(CUDA_EXT)' | sort -k 1nr | cut -f2-)
# Set the object file names, with the source directory stripped
# from the path, and the build path prepended in its place
OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(CUDA_PATH)/%.$(CUDA_EXT)=$(BUILD_CUDA_PATH)/%.o)
# Set the dependency files that will be used to add header dependencies
DEPS = $(OBJECTS:.o=.d)

# flags #
COMPILE_FLAGS = -std=c++11 -Wall -Wextra -g
NVCC_COMPILE_FLAGS = -std=c++11 --default-stream per-thread -g -O3 -arch=$(ARCH)
INCLUDES = -I include/ -I /usr/local/include
# Space-separated pkg-config libraries used by this project
LIBS =

.PHONY: default_target
default_target: release

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS)
release: export NVCCFLAGS := $(EXTRA_CFLAGS) $(NVCC_COMPILE_FLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(dir $(CUDA_OBJECTS))
	@mkdir -p $(BIN_PATH)

.PHONY: clean
clean:
	@echo "Deleting $(BIN_NAME) symlink"
	@$(RM) $(BIN_NAME)
	@echo "Deleting directories"
	@$(RM) -r $(BUILD_PATH)
	@$(RM) -r $(BIN_PATH)

# checks the executable and symlinks to the output
.PHONY: all
all: $(BIN_PATH)/$(BIN_NAME)
	@echo "Making symlink: $(BIN_NAME) -> $<"
	@$(RM) $(BIN_NAME)
	@ln -s $(BIN_PATH)/$(BIN_NAME) $(BIN_NAME)

# Creation of the executable
$(BIN_PATH)/$(BIN_NAME): $(OBJECTS) $(CUDA_OBJECTS)
	@echo "Linking: $@"
	$(NVCC) $(OBJECTS) $(CUDA_OBJECTS) -o $@

# Add dependency files, if they exist
-include $(DEPS)

# Source file rules
# After the first compilation they will be joined with the rules from the
# dependency files to provide header dependencies
$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

$(BUILD_CUDA_PATH)/%.o: $(CUDA_PATH)/%.$(CUDA_EXT)
	@echo "Compiling: $< -> $@"
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

run: dirs $(BIN_PATH)/$(BIN_NAME)
	$(LAUNCHER) $(BIN_PATH)/$(BIN_NAME) ../minimap2-sycl/in-1k.txt kernel-out.txt
