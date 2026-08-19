#===============================================================================
# User Options
#===============================================================================

# Set USE_DOCKER=yes to use PyTorch container, USE_DOCKER=no for local PyTorch
USE_DOCKER = no
LAUNCHER   ?=

#===============================================================================
# PyTorch Configuration
#===============================================================================

# AMD PyTorch container for ROCm support
DOCKER_IMAGE = rocm/pytorch:latest

ifeq ($(USE_DOCKER),yes)
  PYTHON_CMD = docker run --rm --runtime=amd --gpu all -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE) python
else
  PYTHON_CMD = python
endif

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all build clean run

all: run

# PyTorch compiles ROCm extensions on-the-fly during first run
build: setup.py
	$(PYTHON_CMD) setup.py build develop

clean:
	rm -rf *.egg-info *.so \
	torchpairwise/*.so \
	build __pycache__ \
	torchpairwise/__pycache__ \
	torchpairwise/ops/__pycache__ \
	torchpairwise/ops/metrics/__pycache__ \
	torchpairwise/ops/metrics/*.pyi

run: build
	$(LAUNCHER) $(PYTHON_CMD) tests/main.py
