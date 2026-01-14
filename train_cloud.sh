#!/bin/bash
# Cloud training script - runs all agent training
# Run from ~/rlcard_NFL directory

set -e

# Activate environment
source ~/nfl_env/bin/activate
cd ~/rlcard_NFL

echo "============================================"
echo "Starting Cloud Training"
echo "Started at: $(date)"
echo "============================================"

# 1. Train PPO
echo ""
echo "[1/3] Training PPO Agent..."
python examples/run_ppo_nfl.py --game nfl-bucketed --episodes 50000 \
    --cached-model \
    --lr 0.001 --entropy-coef 0.1 \
    --save-dir models/ppo_cloud

# 2. Train DMC (12 actors for 16 vCPU)
echo ""
echo "[2/3] Training DMC Agent..."
python examples/run_dmc_nfl.py --game nfl-bucketed --cached-model \
    --iterations 100000 \
    --num-actors 12 \
    --save-dir experiments/dmc_cloud

# 3. Train NFSP (optional - comment out if not needed)
echo ""
echo "[3/3] Training NFSP Agent..."
python examples/nfl_nfsp_train.py --cached-model --episodes 100000 \
    --anticipatory-param 0.3 \
    --epsilon-decay-steps 10000 \
    --reservoir-capacity 5000 \
    --rl-lr 0.001 --sl-lr 0.001 \
    --save-dir models/nfsp_cloud

echo ""
echo "============================================"
echo "All Training Complete"
echo "Finished at: $(date)"
echo "============================================"
echo ""
echo "Models saved to:"
echo "  - models/ppo_cloud"
echo "  - experiments/dmc_cloud"
echo "  - models/nfsp_cloud"
