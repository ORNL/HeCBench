#===============================================================================
# User Options
#===============================================================================

# Set USE_DOCKER=yes to use PyTorch container, USE_DOCKER=no for local PyTorch
USE_DOCKER = no
LAUNCHER   ?=

#===============================================================================
# PyTorch Configuration
#===============================================================================

# NVIDIA PyTorch container for CUDA support
DOCKER_IMAGE = nvcr.io/nvidia/pytorch:25.09-py3

ifeq ($(USE_DOCKER),yes)
  PYTHON_CMD = docker run --rm --gpus all -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE) python
else
  PYTHON_CMD = python
endif

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all build clean run

all: run

# PyTorch compiles CUDA extensions on-the-fly during first run
build: setup.py
	$(PYTHON_CMD) setup.py build develop

clean:
	rm -rf *.egg-info *.so build __pycache__

run: build
	$(LAUNCHER) $(PYTHON_CMD) main.py
