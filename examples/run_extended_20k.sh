#!/bin/bash

echo "Starting Extended Training"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo --num_episodes 4000000 --resume > logs/ppo_standard_40k.log 2>&1 &
echo "Started PPO (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp --num_episodes 4000000 --resume > logs/nfsp_standard_40k.log 2>&1 &
echo "Started NFSP (Standard) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent deep_cfr --num_episodes 40000 --resume > logs/deep_cfr_standard_40k.log 2>&1 &
echo "Started DeepCFR (Standard) [PID $!] (Target: 20,000 iterations)"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo --num_episodes 4000000 --resume > logs/ppo_iig_40k.log 2>&1 &
echo "Started PPO (IIG) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp --num_episodes 4000000 --resume > logs/nfsp_iig_40k.log 2>&1 &
echo "Started NFSP (IIG) [PID $!]"

nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent deep_cfr --num_episodes 40000 --resume > logs/deep_cfr_iig_40k.log 2>&1 &
echo "Started DeepCFR (IIG) [PID $!] (Target: 20,000 iterations)"

echo "---------------------------------------------------"
echo "All jobs started in background."
echo "You can view progress with: tail -f logs/*_40k.log"
echo "You can safely close this terminal now."
