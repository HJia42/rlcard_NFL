#!/bin/bash
# Setup script for Google Cloud VM
# Run this after SSH'ing into the VM

set -e

echo "============================================"
echo "Setting up NFL RL Training Environment"
echo "============================================"

# Update system
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git unzip

# Create virtual environment
python3 -m venv ~/nfl_env
source ~/nfl_env/bin/activate

# Install PyTorch with CUDA (for T4 GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install numpy pandas scipy scikit-learn tqdm

# Install the project in development mode
cd ~/rlcard_NFL
pip install -e .

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate environment: source ~/nfl_env/bin/activate"
echo "To start training: bash train_cloud.sh"
