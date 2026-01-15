#!/bin/bash
# Setup script for Google Cloud VM with Deep Learning Image
# Run this after SSH'ing into the VM

set -e

echo "============================================"
echo "Setting up NFL RL Training Environment"
echo "============================================"

# Check GPU
echo ""
echo "Checking GPU..."
nvidia-smi || echo "No GPU found"

# Clone project if not exists
if [ ! -d ~/rlcard_NFL ]; then
    echo ""
    echo "Cloning repository..."
    git clone https://github.com/HJia42/rlcard_NFL.git ~/rlcard_NFL
fi

cd ~/rlcard_NFL

# Install dependencies (use existing Python from deep learning image)
echo ""
echo "Installing dependencies..."
pip install -e . -q
pip install cython numpy pandas scipy scikit-learn tqdm -q

# Compile Cython
echo ""
echo "Compiling Cython extensions..."
python setup_cython.py build_ext --inplace || echo "Cython build failed (optional)"

# Verify PyTorch GPU
echo ""
echo "Checking PyTorch CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To start training: bash train_cloud.sh"
echo "Or run individual agents:"
echo "  python examples/run_ppo_nfl.py --device cuda --cached-model --episodes 100000"
echo "  python examples/train_all.py --agents all"
