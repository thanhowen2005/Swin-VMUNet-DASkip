#!/usr/bin/env bash
set -euo pipefail

cd "$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Run in the activated training environment on a CUDA GPU machine.
python -c 'import torch; from torch.utils.cpp_extension import CUDA_HOME; assert torch.cuda.is_available(), "A CUDA-enabled PyTorch environment and visible GPU are required"; assert CUDA_HOME is not None, "CUDA Toolkit (nvcc) is required to build the original extensions"'
python -m pip install packaging ninja wheel

(
    cd models/vmunet/damamba_original/selective_scan
    python -m pip install --no-build-isolation --no-deps .
)
(
    cd models/vmunet/damamba_original/ops_dcnv3
    python -m pip install --no-build-isolation --no-deps .
)

python -c 'import torch, DCNv3, selective_scan_cuda_oflex_rh; from models.vmunet.damamba_original.DAMamba import Block, DASSM; print("Original DAMamba extensions and Block imported successfully")'
