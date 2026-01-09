"""
Train agents on NFL game using RLCard

Available algorithms:
- DQN: Deep Q-Network
- DMC: Deep Monte Carlo (best for card games)
- CFR: Counterfactual Regret Minimization
"""

import sys
sys.path.insert(0, '.')

import argparse
import os

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, tournament


def train_dqn(args):
    """Train DQN agents."""
    env = rlcard.make('nfl', config={'seed': args.seed})
    eval_env = rlcard.make('nfl', config={'seed': args.seed + 1})
    
    device = get_device()
    
    # Create agents
    agents = []
    for i in range(env.num_players):
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=[11],  # Max state dimension
            mlp_layers=[128, 128, 128],
            device=device,
            learning_rate=0.0005,
            batch_size=64,
            update_target_estimator_every=1000,
            replay_memory_size=20000,
        )
        agents.append(agent)
    
    env.set_agents(agents)
    eval_env.set_agents(agents)
    
    print(f"Training DQN on NFL game for {args.num_episodes} episodes...")
    print(f"Device: {device}")
    
    for ep in range(1, args.num_episodes + 1):
        # Train one episode
        trajectories, payoffs = env.run(is_training=True)
        
        # Feed transitions
        for i, agent in enumerate(agents):
            for ts in zip(trajectories[i][:-1], trajectories[i][1:]):
                agent.feed(ts)
        
        # Evaluate periodically
        if ep % args.eval_every == 0:
            # Evaluate against random
            random_agent = RandomAgent(num_actions=env.num_actions)
            
            eval_payoffs = []
            for _ in range(args.num_eval_games):
                eval_env.set_agents([agents[0], random_agent])
                _, p = eval_env.run(is_training=False)
                eval_payoffs.append(p[0])
            
            avg_payoff = sum(eval_payoffs) / len(eval_payoffs)
            print(f"Episode {ep}: Avg Offense vs Random = {avg_payoff:.2f}")
        
        # Save periodically
        if ep % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                save_path = os.path.join(args.save_dir, f'dqn_player_{i}_{ep}.pth')
                agent.save(save_path)
            print(f"Saved models at episode {ep}")
    
    # Final save
    for i, agent in enumerate(agents):
        save_path = os.path.join(args.save_dir, f'dqn_player_{i}_final.pth')
        agent.save(save_path)
    
    print("Training complete!")


def train_cfr(args):
    """Train CFR agent."""
    from rlcard.agents import CFRAgent
    
    env = rlcard.make('nfl', config={'seed': args.seed, 'allow_step_back': True})
    
    agent = CFRAgent(env, model_path=os.path.join(args.save_dir, 'cfr'))
    agent.train_episodes = args.num_episodes
    
    print(f"Training CFR on NFL game for {args.num_episodes} iterations...")
    
    for ep in range(1, args.num_episodes + 1):
        agent.train()
        
        if ep % args.eval_every == 0:
            print(f"Iteration {ep}: Training...")
        
        if ep % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            agent.save()
            print(f"Saved CFR model at iteration {ep}")
    
    agent.save()
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description='Train agents on NFL')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'cfr'])
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--num_eval_games', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='models/nfl')
    
    args = parser.parse_args()
    
    if args.algorithm == 'dqn':
        train_dqn(args)
    elif args.algorithm == 'cfr':
        train_cfr(args)


if __name__ == '__main__':
    main()
