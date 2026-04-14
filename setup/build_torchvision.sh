#!/bin/bash
set -e
source move-one/bin/activate

echo "1. Cleaning up and uninstalling torchvision..."
# Force uninstall from venv and user site-packages to be sure
uv pip uninstall torchvision --python move-one/bin/python || true
pip uninstall -y torchvision || true

echo "2. Preparing torchvision v0.20.1 source (Full Clean)..."
if [ ! -d "torchvision_repo" ]; then
    git clone --branch v0.20.1 https://github.com/pytorch/vision torchvision_repo
fi
cd torchvision_repo
rm -rf build/ dist/ torchvision.egg-info/

echo "3. Building & Installing with CUDA support (using uv)..."
export BUILD_VERSION=0.20.1
export FORCE_CUDA=1
export CUDA_HOME=/usr/local/cuda
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

# Re-activate just in case
source ../move-one/bin/activate

# Build wheel
python setup.py bdist_wheel

# Install within the venv using uv
uv pip install dist/*.whl --python ../move-one/bin/python --no-cache

echo "4. Verification..."
cd ..
python -c "import torch; import torchvision; print(f'Torch: {torch.__version__}'); print(f'TorchVision: {torchvision.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Torchvision C extension: {torchvision._C if hasattr(torchvision, \"_C\") else \"Missing\"}')"
python -c "from torchvision.ops import nms; print('Successfully imported CUDA NMS ops')"
