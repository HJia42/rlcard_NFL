"""
Standardized Training Script for NFL RL Agents

Trains all agents (PPO, DMC, NFSP, Deep CFR) with normalized "total games"
to enable fair comparison in academic papers.

Game Equivalences:
- PPO: 1 episode = 1 game
- NFSP: 1 episode = 1 game
- DMC: ~10 games per 3200 frames (batch_size=32, unroll=100)
- Deep CFR: ~100 games per iteration (traversals * samples)

Usage:
    python examples/train_all_standardized.py --total-games 100000
    python examples/train_all_standardized.py --total-games 500000 --agents ppo dmc
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Conversion factors: how many "games" per unit
GAMES_PER_UNIT = {
    'ppo': 1,           # 1 episode = 1 game
    'nfsp': 1,          # 1 episode = 1 game
    'dmc': 0.01,        # ~100 frames = 1 game (conservative)
    'deep_cfr': 100,    # 1 iteration ≈ 100 games (traversals)
}


def games_to_iterations(agent: str, total_games: int) -> int:
    """Convert total games to agent-specific iterations."""
    factor = GAMES_PER_UNIT.get(agent, 1)
    return int(total_games / factor)


def run_agent(agent: str, total_games: int, save_dir: str, device: str, game: str, extra_args: list = None):
    """Run a single agent's training."""
    
    iterations = games_to_iterations(agent, total_games)
    
    cmd = ['python3']
    
    if agent == 'ppo':
        cmd += [
            'examples/run_ppo_nfl.py',
            '--game', game,
            '--cached-model',
            '--episodes', str(iterations),
            '--device', device,
            '--save-dir', f'{save_dir}/ppo',
        ]
    
    elif agent == 'dmc':
        cuda_arg = '0' if device == 'cuda' else ''
        cmd += [
            'examples/run_dmc_nfl.py',
            '--game', game,
            '--cached-model',
            '--iterations', str(iterations),
            '--num-actors', '4',
            '--cuda', cuda_arg,
            '--save-dir', f'{save_dir}/dmc',
        ]
    
    elif agent == 'nfsp':
        cmd += [
            'examples/nfl_nfsp_train.py',
            '--game', game,
            '--cached-model',
            '--episodes', str(iterations),
            '--device', device,
            '--save-dir', f'{save_dir}/nfsp',
        ]
    
    elif agent == 'deep_cfr':
        cmd += [
            'examples/run_deep_cfr_nfl.py',
            '--game', game,
            '--iterations', str(iterations),
            '--model_path', f'{save_dir}/deep_cfr',
            '--device', device,
        ]
    
    else:
        print(f"Unknown agent: {agent}")
        return False
    
    if extra_args:
        cmd += extra_args
    
    print(f"\n{'='*60}")
    print(f"Training {agent.upper()}")
    print(f"{'='*60}")
    print(f"Game: {game}")
    print(f"Total games equivalent: {total_games:,}")
    print(f"Agent iterations: {iterations:,}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    elapsed = time.time() - start_time
    
    print(f"\n{agent.upper()} completed in {elapsed/60:.1f} minutes")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Standardized NFL RL Training')
    parser.add_argument('--total-games', type=int, default=100000,
                        help='Total games to train (normalized across agents)')
    parser.add_argument('--agents', nargs='+', default=['ppo', 'dmc', 'nfsp', 'deep_cfr'],
                        choices=['ppo', 'dmc', 'nfsp', 'deep_cfr', 'all'],
                        help='Agents to train')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed', 'nfl-iig', 'nfl-iig-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--save-dir', type=str, default='models/paper',
                        help='Base directory for saving models')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for training')
    parser.add_argument('--sequential', action='store_true',
                        help='Train agents sequentially (default)')
    
    args = parser.parse_args()
    
    if 'all' in args.agents:
        agents = ['ppo', 'dmc', 'nfsp', 'deep_cfr']
    else:
        agents = args.agents
    
    print("="*60)
    print("STANDARDIZED NFL RL TRAINING")
    print("="*60)
    print(f"Game: {args.game}")
    print(f"Total games per agent: {args.total_games:,}")
    print(f"Agents: {', '.join(agents)}")
    print(f"Device: {args.device}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    
    print("\nIteration equivalences:")
    for agent in agents:
        iters = games_to_iterations(agent, args.total_games)
        print(f"  {agent.upper()}: {iters:,} iterations")
    
    print("\n")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    results = {}
    total_start = time.time()
    
    for agent in agents:
        success = run_agent(agent, args.total_games, args.save_dir, args.device, args.game)
        results[agent] = 'SUCCESS' if success else 'FAILED'
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"\nResults:")
    for agent, status in results.items():
        print(f"  {agent.upper()}: {status}")
    print("="*60)


if __name__ == '__main__':
    main()
