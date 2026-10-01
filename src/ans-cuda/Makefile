ARCH = sm_60

NVCC = nvcc
NVCC_ARCH = -gencode arch=compute_$(ARCH:sm_%=%),code=$(ARCH)
NVCC_FLAGS = --std=c++17 -O3

INC_DIR = include
SRC_DIR = src
OBJ_DIR = bin
EXEC_NAME = main

SRC_FILES := $(wildcard $(SRC_DIR)/*.cc)
CU_SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES := $(patsubst $(SRC_DIR)/%.cc,$(OBJ_DIR)/%.o,$(SRC_FILES))
CU_OBJ_FILES := $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.obj,$(CU_SRC_FILES))

$(OBJ_DIR)/$(EXEC_NAME): $(OBJ_FILES) $(CU_OBJ_FILES)
	$(NVCC) $(OBJ_FILES) $(CU_OBJ_FILES) -o $(OBJ_DIR)/$(EXEC_NAME)

$(OBJ_DIR)/%.obj: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_ARCH) $(NVCC_FLAGS) -I $(INC_DIR) -c -o $@ $<

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cc
	$(NVCC) $(NVCC_ARCH) $(NVCC_FLAGS) -I $(INC_DIR) -c -o $@ $<

.PHONY: clean
clean:
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.o
	rm -f $(RM_FLAGS) $(OBJ_DIR)/*.obj
	rm -f $(RM_FLAGS) $(OBJ_DIR)/$(EXEC_NAME)

run: $(OBJ_DIR)/$(EXEC_NAME)
	$(LAUNCHER) ./$(OBJ_DIR)/$(EXEC_NAME) 10
