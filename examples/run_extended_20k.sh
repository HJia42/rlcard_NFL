#!/bin/bash

echo "Starting Extended Training (Version 3 - Fix Std Env - Parallel)"

# Parallel execution with reduced episodes for Speed
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

# Standard Environment (Now fixes dimensions)
nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo --num_episodes 100000 --reward_type score --full_game --log_dir experiments/nfl_score_full_ppo_std_v3 --resume > logs/ppo_standard_score_full_100k_v3.log 2>&1 &
echo "Started PPO (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp --num_episodes 100000 --reward_type score --full_game --log_dir experiments/nfl_score_full_nfsp_std_v3 --resume > logs/nfsp_standard_score_full_100k_v3.log 2>&1 &
echo "Started NFSP (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent deep_cfr --num_episodes 10000 --reward_type score --full_game --log_dir experiments/nfl_score_full_deep_cfr_std_v3 --resume > logs/deep_cfr_standard_score_full_10k_v3.log 2>&1 &
echo "Started DeepCFR (Standard) [PID $!]"

# IIG Environment
nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo --num_episodes 100000 --reward_type score --full_game --log_dir experiments/nfl_score_full_ppo_iig_v3 --resume > logs/ppo_iig_score_full_100k_v3.log 2>&1 &
echo "Started PPO (IIG) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp --num_episodes 100000 --reward_type score --full_game --log_dir experiments/nfl_score_full_nfsp_iig_v3 --resume > logs/nfsp_iig_score_full_100k_v3.log 2>&1 &
echo "Started NFSP (IIG) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent deep_cfr --num_episodes 10000 --reward_type score --full_game --log_dir experiments/nfl_score_full_deep_cfr_iig_v3 --resume > logs/deep_cfr_iig_score_full_10k_v3.log 2>&1 &
echo "Started DeepCFR (IIG) [PID $!]"

echo "---------------------------------------------------"
echo "All jobs started in background."
echo "You can view progress with: tail -f logs/*_v3.log"
echo "You can safely close this terminal now."
