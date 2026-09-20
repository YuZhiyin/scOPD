# Installation Instructions

This guide provides instructions for setting up the environment to run the SDPO codebase.

## System Requirements
*   **Operating System:** Linux (Tested on SLES 15 SP5 and Ubuntu 22.04)
*   **Hardware:** NVIDIA GPUs (CUDA compatible)
*   **Python:** 3.12 (Tested on 3.12.3)
*   **CUDA Driver:** Compatible with the PyTorch version installed (see below).

## 1. Core Installation

Choose **one** of the following methods to set up your environment.

### Method A: Local Python Environment (Recommended)
This is the standard approach for local workstations (e.g., RTX 5090).

**1. Install PyTorch:**
```bash
# Install PyTorch 2.5.1 (Stable for CUDA 12.4)
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**2. Install SDPO and Dependences:**
From the root of the repository:
```bash
# Install dependencies
# Option 1: Core pinned dependencies (Recommended)
pip install -r requirements.txt

# Option 2: Exact tested environment
# pip install -r requirements/full.txt

# Install SDPO (verl) in editable mode
pip install -e .


# Install Flash Attention 2
pip install flash-attn --no-build-isolation
```

---

### Method B: Docker (Stable & Reproducible)
Use this if you want a guaranteed working environment without managing local dependencies.

**1. Build and Run:**
```bash
# Build the image
docker build -f docker/Dockerfile.sdpo -t sdpo:latest .

# Run container (with GPU support)
docker run --gpus all -it --ipc=host -v $(pwd):/app sdpo:latest
```
*Inside the container, SDPO is already installed and ready to use.*


> [!NOTE]
> For more specific instructions on `verl` architecture and advanced configuration, refer to the [official verl repository](https://github.com/volcengine/verl).

## 2. Advanced / Optional Components

These components are not strictly required for the basic PPO training loop but are needed for specific advanced workflows.

### vLLM & SGLang (High-Performance Inference)

This codebase supports vLLM and SGLang for high-throughput inference, which significantly accelerates the rollout phase of reinforcement learning. While optional for basic usage, they are recommended for large-scale training.

**Installation:**
```bash
pip install -r requirements/sglang.txt
```
*Note: This command installs specific versions of SGLang and vLLM compatible with this codebase. Ensure your NVIDIA drivers are compatible with the installed CUDA toolkit (e.g., CUDA 12.4 if matching the PyTorch installation above).*

# Verification
To verify the installation, you can run the tests:

```bash
# Install test dependencies
pip install -r requirements/test.txt

# Run tests
pytest tests/
```

---

## Appendix: Development Environment Reference
This codebase was developed and tested using the **NVIDIA NGC 25.12** software stack. While we recommend stable releases for general use, the exact environment state is:

- **PyTorch**: `2.10.0a0+b4e4ee81d3.nv25.12`
- **NGC Index**: `https://pypi.ngc.nvidia.com`
- **CUDA**: 12.x (Optimized for GH200/H100)
