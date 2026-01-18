#!/bin/bash

# Extended Training Script (20,000 Episodes)
# Uses nohup to run in background so you can close the terminal.

echo "Starting Extended Training (Target: 20,000 Episodes)..."

# 1. PPO (Standard)
# Force 1 thread per job to avoid locking up the VM
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TORCH_NUM_THREADS=1

nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo --num_episodes 20000 --resume > logs/ppo_standard_20k.log 2>&1 &
echo "Started PPO (Standard) [PID $!]"

# 2. NFSP (Standard)
nohup python3 examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp --num_episodes 20000 --resume > logs/nfsp_standard_20k.log 2>&1 &
echo "Started NFSP (Standard) [PID $!]"

# 3. PPO (IIG)
nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo --num_episodes 20000 --resume > logs/ppo_iig_20k.log 2>&1 &
echo "Started PPO (IIG) [PID $!]"

# 4. NFSP (IIG)
nohup python3 examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp --num_episodes 20000 --resume > logs/nfsp_iig_20k.log 2>&1 &
echo "Started NFSP (IIG) [PID $!]"

echo "---------------------------------------------------"
echo "All jobs started in background."
echo "You can view progress with: tail -f logs/*_20k.log"
echo "You can safely close this terminal now."
