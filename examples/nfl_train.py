"""
Train agents on NFL game using RLCard

Usage:
    python examples/nfl_train.py --num_episodes 10000
"""

import sys
sys.path.insert(0, '.')

import argparse
import os

import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import get_device, reorganize
from rlcard.utils.eval_utils import quick_eval, format_eval_line, EvalLogger


def train_dqn(args):
    """Train DQN agents via self-play."""
    env = rlcard.make(args.game, config={'seed': args.seed})
    
    device = get_device()
    
    # Create DQN agents for both players
    agents = []
    for i in range(env.num_players):
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=[11],
            mlp_layers=[128, 128, 128],
            device=device,
            learning_rate=0.0005,
            batch_size=64,
            update_target_estimator_every=1000,
            replay_memory_size=20000,
        )
        agents.append(agent)
    
    env.set_agents(agents)
    # Evaluation uses standardized utils (creates its own env)
    
    print(f"Training DQN on NFL game for {args.num_episodes} episodes...")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, 'eval_log.csv')
    eval_logger = EvalLogger(log_path)
    
    for ep in range(1, args.num_episodes + 1):
        # Run one episode
        trajectories, payoffs = env.run(is_training=True)
        
        # Reorganize trajectories into proper format
        trajectories = reorganize(trajectories, payoffs)
        
        # Feed transitions to agents
        for i, agent in enumerate(agents):
            for transition in trajectories[i]:
                agent.feed(transition)
        
        # Evaluate periodically
        if ep % args.eval_every == 0:
            results = quick_eval(agents[0], args.game, num_games=args.num_eval_games)
            eval_logger.log(ep, results)
            print(format_eval_line(ep, results))
        
        # Save periodically
        if ep % args.save_every == 0:
            for i, agent in enumerate(agents):
                agent.save_checkpoint(args.save_dir, filename=f'dqn_player_{i}_{ep}.pt')
            print(f"Saved models at episode {ep}")
    
    # Final save
    for i, agent in enumerate(agents):
        agent.save_checkpoint(args.save_dir, filename=f'dqn_player_{i}_final.pt')
    
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train DQN agents on NFL')
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--num_eval_games', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='models/nfl')
    parser.add_argument('--game', type=str, default='nfl',
                        choices=['nfl', 'nfl-bucketed'],
                        help='Game environment')
    
    args = parser.parse_args()
    train_dqn(args)


if __name__ == '__main__':
    main()
