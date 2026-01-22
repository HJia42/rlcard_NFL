#!/bin/bash

echo "Starting Extended Training"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_ppo_std --resume > logs/ppo_standard_score_full_500k.log 2>&1 &
echo "Started PPO (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_nfsp_std --resume > logs/nfsp_standard_score_full_500k.log 2>&1 &
echo "Started NFSP (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent deep_cfr --num_episodes 50000 --reward_type score --full_game --log_dir experiments/nfl_score_full_deep_cfr_std --resume > logs/deep_cfr_standard_score_full_50k.log 2>&1 &
echo "Started DeepCFR (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_ppo_iig --resume > logs/ppo_iig_score_full_500k.log 2>&1 &
echo "Started PPO (IIG) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp --num_episodes 500000 --reward_type score --full_game --log_dir experiments/nfl_score_full_nfsp_iig --resume > logs/nfsp_iig_score_full_500k.log 2>&1 &
echo "Started NFSP (IIG) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent deep_cfr --num_episodes 50000 --reward_type score --full_game --log_dir experiments/nfl_score_full_deep_cfr_iig --resume > logs/deep_cfr_iig_score_full_50k.log 2>&1 &
echo "Started DeepCFR (IIG) [PID $!]"

echo "---------------------------------------------------"
echo "All jobs started in background."
echo "You can view progress with: tail -f logs/*_40k.log"
echo "You can safely close this terminal now."
