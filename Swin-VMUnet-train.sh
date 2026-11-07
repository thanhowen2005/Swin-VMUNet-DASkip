#!/bin/bash
#SBATCH --job-name=train_bs32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00

set -euo pipefail
module purge
eval "$(conda shell.bash hook)"
conda activate /media/nnthao21/venvs/vmunet

export LD_LIBRARY_PATH="$(dirname "$(ldconfig -p | awk '$1 == "libcuda.so.1" && !found {print $NF; found=1}')")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_HOME=$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")
TORCH_LIB=$(python -c 'import os, torch; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')
export LD_LIBRARY_PATH="$TORCH_LIB:$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST=$(python -c 'import torch; a,b=torch.cuda.get_device_capability(0); print(f"{a}.{b}")')

cd /media/nnthao21/models/Swin-VMUNet-DASkip

# Rebuild on the allocated GPU; discard objects built for previous GPUs.
for extension in selective_scan ops_dcnv3; do
    (
        cd "models/vmunet/damamba_original/$extension"
        python setup.py clean --all
        python -m pip install --no-build-isolation --no-deps --force-reinstall .
    )
done

python -u train.py
