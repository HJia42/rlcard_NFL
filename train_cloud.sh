#!/bin/bash
# Cloud training script - runs all agent training
# Run from ~/rlcard_NFL directory

set -e

# Activate environment (skip if using deep learning image)
if [ -d ~/nfl_env ]; then
    source ~/nfl_env/bin/activate
fi
cd ~/rlcard_NFL

echo "============================================"
echo "Starting Cloud Training"
echo "Started at: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'None')"
echo "============================================"

# Compile Cython if available
if [ -f setup_cython.py ]; then
    echo ""
    echo "[Setup] Compiling Cython..."
    pip install cython numpy -q
    python3 setup_cython.py build_ext --inplace 2>/dev/null || echo "Cython build skipped"
fi

# 1. Train PPO (GPU)
echo ""
echo "[1/4] Training PPO Agent..."
python3 examples/run_ppo_nfl.py --game nfl-bucketed --episodes 100000 \
    --cached-model \
    --lr 0.001 --entropy-coef 0.1 \
    --device cuda \
    --save-dir models/ppo_cloud

# 2. Train DMC (GPU, 4 actors for memory efficiency)
echo ""
echo "[2/4] Training DMC Agent..."
python3 examples/run_dmc_nfl.py --game nfl-bucketed --cached-model \
    --num-episodes 500000 \
    --num-actors 4 \
    --cuda 0 \
    --save-dir models/dmc_cloud

# 3. Train NFSP (GPU)
echo ""
echo "[3/4] Training NFSP Agent..."
python3 examples/nfl_nfsp_train.py --cached-model --episodes 500000 \
    --anticipatory-param 0.3 \
    --epsilon-decay-steps 10000 \
    --reservoir-capacity 5000 \
    --rl-lr 0.001 --sl-lr 0.001 \
    --device cuda \
    --save-dir models/nfsp_cloud

# 4. Train MCCFR (CPU, benefits from Cython)
echo ""
echo "[4/4] Training MCCFR Agent..."
python3 examples/run_mccfr.py --iterations 50000 \
    --model_path models/mccfr_cloud

echo ""
echo "============================================"
echo "All Training Complete"
echo "Finished at: $(date)"
echo "============================================"
echo ""
echo "Models saved to:"
echo "  - models/ppo_cloud"
echo "  - models/dmc_cloud"
echo "  - models/nfsp_cloud"
echo "  - models/mccfr_cloud"
