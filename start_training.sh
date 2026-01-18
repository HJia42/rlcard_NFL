#!/bin/bash

# Force PyTorch to use only 1 thread per process to prevent CPU saturation
# When running 6 jobs on 8 vCPUs, allowing default parallelism causes huge contention.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Starting training with OMP_NUM_THREADS=1 to save CPU..."

# Create logs directory if not exists
mkdir -p logs

# Standard Environment (nfl-bucketed)
nohup python examples/train_bucketed_agents.py --env nfl-bucketed --agent ppo > logs/ppo_std.log 2>&1 &
echo "Started PPO (Standard) - PID $!"

nohup python examples/train_bucketed_agents.py --env nfl-bucketed --agent nfsp > logs/nfsp_std.log 2>&1 &
echo "Started NFSP (Standard) - PID $!"

nohup python examples/train_bucketed_agents.py --env nfl-bucketed --agent deep_cfr > logs/cfr_std.log 2>&1 &
echo "Started DeepCFR (Standard) - PID $!"

# Imperfect Information Environment (nfl-iig-bucketed)
nohup python examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent ppo > logs/ppo_iig.log 2>&1 &
echo "Started PPO (IIG) - PID $!"

nohup python examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent nfsp > logs/nfsp_iig.log 2>&1 &
echo "Started NFSP (IIG) - PID $!"

nohup python examples/train_bucketed_agents.py --env nfl-iig-bucketed --agent deep_cfr > logs/cfr_iig.log 2>&1 &
echo "Started DeepCFR (IIG) - PID $!"

echo "All 6 jobs running in background."
echo "Monitor with: tail -f logs/*.log"
