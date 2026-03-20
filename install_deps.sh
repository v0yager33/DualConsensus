#!/bin/bash
set -e
export MAX_JOBS=32

echo "1. Install inference frameworks and PyTorch"
pip install --no-cache-dir "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict<=0.6.2" torchdata
pip install --no-cache-dir vllm

echo "2. Install basic packages"
pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=19.0.0" pandas \
    ray[default] codetiming hydra-core pylatexenc wandb dill pybind11 liger-kernel \
    pytest pre-commit ruff

pip install "packaging>=20.0" uvicorn fastapi \
    "nvidia-ml-py>=12.560.30" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

echo "3. Install FlashAttention and FlashInfer"
# Install flash-attn-2.7.4.post1 (cxx11abi=False)
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Install flashinfer-0.2.2.post1+cu124 (cxx11abi=False)
wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    pip install --no-cache-dir flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

echo "4. May need to fix opencv"
pip install opencv-python
pip install opencv-fixer && \
    python -c "from opencv_fixer import AutoFix; AutoFix()"

echo "5. Install math verification deps"
pip install latex2sympy2_extended math-verify

echo "6. Install verl"
pip install -e ./verl

echo "Successfully installed all packages"
