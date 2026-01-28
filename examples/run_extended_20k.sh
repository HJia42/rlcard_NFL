#!/bin/bash

echo "Starting Extended Training (Version 3 - Fix Std Env)"

# Sequential execution to prevent OOM
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# Standard Environment (Now fixes dimensions)
echo "---------------------------------------------------"
echo "Starting PPO (Standard)..."
python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_ppo_std_v3 --resume > logs/ppo_standard_score_full_500k_v3.log 2>&1

echo "Starting NFSP (Standard)..."
python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_nfsp_std_v3 --resume > logs/nfsp_standard_score_full_500k_v3.log 2>&1

echo "Starting DeepCFR (Standard)..."
python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent deep_cfr --num_episodes 50000 --reward_type score --full_game --log_dir experiments/nfl_score_full_deep_cfr_std_v3 --resume > logs/deep_cfr_standard_score_full_50k_v3.log 2>&1


# IIG Environment (Control Group - should match prev performance but good to retrain for consistency)
echo "---------------------------------------------------"
echo "Starting PPO (IIG)..."
python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_ppo_iig_v3 --resume > logs/ppo_iig_score_full_500k_v3.log 2>&1

echo "Starting NFSP (IIG)..."
python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_nfsp_iig_v3 --resume > logs/nfsp_iig_score_full_500k_v3.log 2>&1

echo "Starting DeepCFR (IIG)..."
python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent deep_cfr --num_episodes 50000 --reward_type score --full_game --log_dir experiments/nfl_score_full_deep_cfr_iig_v3 --resume > logs/deep_cfr_iig_score_full_50k_v3.log 2>&1

echo "---------------------------------------------------"
echo "All jobs completed."
