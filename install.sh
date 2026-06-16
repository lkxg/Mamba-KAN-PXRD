#!/bin/bash
set -e

echo "=== Mamba-KAN-PXRD 环境安装脚本 ==="
echo ""

# 1. 添加 NVIDIA CUDA 仓库
echo "[1/5] 添加 NVIDIA CUDA 仓库..."
if [ ! -f /etc/apt/sources.list.d/cuda-* ]; then
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    sudo apt update
fi

# 2. 安装 CUDA Toolkit 12.8
echo "[2/5] 安装 CUDA Toolkit 12.8..."
if [ ! -f /usr/local/cuda-12.8/bin/nvcc ]; then
    sudo apt install -y cuda-toolkit-12-8
fi

# 3. 设置环境变量
export PATH=/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8

# 4. 安装核心依赖
echo "[3/5] 安装核心 Python 依赖..."
pip install --quiet torch==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install --quiet \
    "numpy>=2.0,<2.3" \
    "pandas>=2.2,<2.4" \
    "scikit-learn>=1.5,<1.8" \
    "scipy>=1.13,<1.16" \
    "pyyaml>=6.0.2,<7" \
    "tqdm>=4.66,<5" \
    "boto3>=1.34,<2" \
    "requests>=2.32,<3" \
    "matplotlib>=3.8,<3.11" \
    "wandb>=0.17,<1" \
    "setuptools>=61.0.0,<82"

# 5. 安装 Mamba 相关库
echo "[4/5] 编译安装 mamba-ssm 和 causal-conv1d..."
pip install --quiet --no-build-isolation causal-conv1d>=1.2 mamba-ssm>=2.2

# 6. 验证安装
echo "[5/5] 验证安装..."
python3 -c "
import torch
import mamba_ssm
import causal_conv1d
import numpy
import pandas
import sklearn
print('✓ 所有依赖安装成功！')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  Mamba SSM: {mamba_ssm.__version__}')
print(f'  Causal Conv1d: {causal_conv1d.__version__}')
"

echo ""
echo "=== 安装完成 ==="
