# Merlin Setup Guide

Complete setup instructions for installing and configuring Merlin.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Steps](#installation-steps)
3. [Verification](#verification)
4. [Troubleshooting](#troubleshooting)
5. [GPU Setup](#gpu-setup)

---

## System Requirements

### Minimum Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11, Linux, macOS |
| Python | 3.9, 3.10, 3.11 |
| RAM | 16 GB minimum, 32 GB recommended |
| Storage | 20 GB (models + data) |
| GPU | Optional but recommended (NVIDIA CUDA) |

### Recommended Setup

| Component | Recommendation |
|-----------|-----------------|
| GPU | NVIDIA RTX 3090/4090 or better |
| CUDA | 11.8+ or 12.1+ |
| cuDNN | 8.6+ |
| Python | 3.10 or 3.11 |
| RAM | 32 GB |
| Storage | SSD with 50 GB+ free space |

---

## Installation Steps

### Step 1: Clone Repository

```bash
# Using Git
git clone https://github.com/sushilb404/Merlin.git
cd Merlin
```

### Step 2: Set Up Python Environment

#### Option A: Conda (Recommended for GPU)

```bash
# Create environment
conda create -n merlin python=3.10 -y
conda activate merlin

# Install PyTorch with CUDA
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

#### Option B: venv (Standard Python)

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
.\venv\Scripts\activate.ps1
# Linux/Mac:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools
```

### Step 3: Install Merlin Package

```bash
# Install in development mode (editable)
pip install -e .

# Or standard install
pip install .
```

### Step 4: Install Dependencies

```bash
# Core dependencies
pip install torch torchvision transformers

# Medical imaging
pip install monai nibabel pydicom

# Data processing
pip install pandas numpy scipy

# Utilities
pip install tqdm rich pyyaml

# Optional: For advanced features
pip install pytorch-lightning tensorboard wandb
```

### Step 5: Download Pre-trained Model

```bash
# Download model checkpoint
python -c "from merlin.models import load_model; model = load_model(); print('Model loaded successfully!')"
```

The checkpoint will be saved to: `~/.cache/merlin/` or `merlin/models/checkpoints/`

---

## Verification

### 1. Verify Installation

```bash
# Check Python version
python --version

# Verify PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Verify Merlin
python -c "import merlin; print(f'Merlin: {merlin.__version__}')"
```

**Expected Output**:
```
Python 3.10.X
PyTorch: 2.0.X+cu11X
CUDA available: True
Merlin: 1.0.0
```

### 2. Test with Sample Data

```bash
# Run test inference
python documentation/run_on_custom_data.py --help

# Run on sample case (if available)
python documentation/run_on_custom_data.py --case_path ./sample_data/Case_001
```

### 3. Check GPU

```bash
# List available GPUs
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

# Check VRAM
python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
```

---

## Troubleshooting

### Issue: Module not found errors

**Solution**: Reinstall in development mode
```bash
pip install -e . --force-reinstall
```

### Issue: PyTorch not found

**Solution**: Reinstall PyTorch with correct CUDA version
```bash
# For CUDA 11.8
pip install torch torchvision transformers --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision transformers --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision transformers --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Model download fails

**Solution**: Manual download
```bash
# Download from HuggingFace
# https://huggingface.co/sushilb404/merlin-model/

# Or use direct download
wget https://huggingface.co/sushilb404/merlin-model/resolve/main/checkpoint.pt
mkdir -p ~/.cache/merlin/
mv checkpoint.pt ~/.cache/merlin/
```

### Issue: CUDA out of memory

**Solution**: 
```bash
# Reduce batch size in code
# Or use CPU mode
export CUDA_VISIBLE_DEVICES="" 
python script.py
```

### Issue: Import errors on Windows

**Solution**: Use PowerShell with full path
```powershell
.\venv\Scripts\Activate.ps1
python -m pip list
```

---

## GPU Setup

### NVIDIA GPU Support

#### 1. Install NVIDIA Drivers

- Download from: https://www.nvidia.com/Download/driverDetails.aspx
- Select your GPU model and OS
- Install and restart

#### 2. Verify NVIDIA Installation

```bash
# Check driver
nvidia-smi

# Expected output:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 535.XX.XX    Driver Version: 535.XX.XX   CUDA Version: 12.2     |
# +-----------------------------------------------------------------------------+
```

#### 3. Install CUDA Toolkit (if needed)

```bash
# Option A: Conda (Recommended)
conda install cuda-toolkit -c nvidia -y

# Option B: Manual download
# https://developer.nvidia.com/cuda-toolkit
```

#### 4. Verify CUDA from Python

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Count: {torch.cuda.device_count()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Quick benchmark
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
for _ in range(100):
    z = torch.matmul(x, y)
print("GPU test passed!")
```

### Multi-GPU Support

```python
from merlin.models import load_model
import torch

# Automatic multi-GPU (DataParallel)
model = load_model(device='cuda')
model = torch.nn.DataParallel(model)

# Specify GPUs
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
```

---

## Project Structure

After installation, your directory will look like:

```
Merlin/
├── merlin/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── build.py
│   │   ├── load.py
│   │   ├── i3res.py
│   │   ├── radiology_report_generation.py
│   │   └── checkpoints/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataloaders.py
│   │   └── monai_transforms.py
│   └── utils/
│       ├── __init__.py
│       └── huggingface_download.py
├── documentation/
│   ├── 00_START_HERE.md          # Main guide
│   ├── SETUP_GUIDE.md            # This file
│   ├── README_CUSTOM_DATA.md     # Custom data guide
│   ├── run_merlin_analysis.py    # Analysis script
│   └── ...
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## Next Steps

1. ✅ Complete setup using steps above
2. ✅ Verify installation with verification steps
3. ✅ Read `00_START_HERE.md` for quick start
4. ✅ Prepare your dataset following `README_CUSTOM_DATA.md`
5. ✅ Run `python documentation/run_merlin_analysis.py <your_data>`

---

## Environment Variables

### Useful Settings

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Disable GPU
export CUDA_VISIBLE_DEVICES=""

# Set number of threads
export OMP_NUM_THREADS=8

# Reduce verbosity
export TF_CPP_MIN_LOG_LEVEL=2
```

### Windows PowerShell

```powershell
# Set environment variables
$env:CUDA_VISIBLE_DEVICES = "0"
$env:OMP_NUM_THREADS = "8"

# Verify
$env:CUDA_VISIBLE_DEVICES
```

---

## Version Information

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10+ | Tested with 3.10, 3.11 |
| PyTorch | 2.0+ | 2.1.2 recommended |
| CUDA | 11.8+, 12.1+ | Optional, for GPU |
| MONAI | 1.3+ | Medical imaging toolkit |

---

## Uninstall

```bash
# Remove virtual environment
rm -rf venv  # Linux/Mac
rmdir /s venv  # Windows

# Or with conda
conda remove -n merlin --all

# Or pip
pip uninstall merlin
```

---

## Support

For setup issues:
1. Check troubleshooting section above
2. Review GitHub issues: [sushilb404/Merlin](https://github.com/sushilb404/Merlin)
3. Check NVIDIA CUDA documentation: [NVIDIA Docs](https://docs.nvidia.com/cuda/)

---

**Last Updated**: December 5, 2025
**Tested On**: Windows 11, Linux Ubuntu 22.04
