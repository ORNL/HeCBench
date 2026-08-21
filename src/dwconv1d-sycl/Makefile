#===============================================================================
# User Options
#===============================================================================

# Set USE_DOCKER=yes to use PyTorch container, USE_DOCKER=no for local PyTorch
USE_DOCKER = no
LAUNCHER   ?=

#===============================================================================
# PyTorch Configuration
#===============================================================================

# PyTorch container for XPU support
DOCKER_IMAGE = intel/intel-extension-for-pytorch:2.8.10-xpu

ifeq ($(USE_DOCKER),yes)
  PYTHON_CMD = docker run --rm --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
	--ipc=host -v $(PWD):/workspace -w /workspace $(DOCKER_IMAGE) python
else
  PYTHON_CMD = python
endif

#===============================================================================
# Targets to Build
#===============================================================================

.PHONY: all build clean run

all: run

# PyTorch compiles XPU extensions on-the-fly during first run
build:
	@echo "Note: PyTorch will compile XPU extensions on first run"

clean:
	rm -rf __pycache__ build *.so

run:
	$(LAUNCHER) $(PYTHON_CMD) run.py
