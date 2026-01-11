"""Train CFR agent on NFL game

CFR (Counterfactual Regret Minimization) converges to Nash equilibrium.
Requires allow_step_back=True for the environment.

Usage:
    python examples/nfl_cfr_train.py
"""

import sys
sys.path.insert(0, '.')

import os
import argparse

import rlcard
from rlcard.agents import CFRAgent, RandomAgent
from rlcard.utils import set_seed, tournament, Logger, plot_curve


def train(args):
    # Create environments with step_back enabled for CFR
    env = rlcard.make(args.game, config={
        'seed': args.seed,
        'allow_step_back': True,
    })
    
    eval_env = rlcard.make(args.game, config={'seed': args.seed})
    
    set_seed(args.seed)
    
    # Initialize CFR Agent
    agent = CFRAgent(
        env,
        os.path.join(args.log_dir, 'cfr_model'),
    )
    
    # Try to load existing model
    try:
        agent.load()
        print("Loaded existing CFR model")
    except:
        print("Starting fresh CFR training")
    
    # Evaluate against random
    eval_env.set_agents([
        agent,
        RandomAgent(num_actions=env.num_actions),
    ])
    
    print(f"\nTraining CFR on NFL for {args.num_episodes} iterations...")
    
    # Training loop
    with Logger(args.log_dir) as logger:
        for episode in range(args.num_episodes):
            agent.train()
            print(f'\rIteration {episode}', end='')
            
            if episode % args.evaluate_every == 0:
                agent.save()
                result = tournament(eval_env, args.num_eval_games)[0]
                logger.log_performance(episode, result)
                print(f'\nEpisode {episode}: Offense vs Random = {result:.2f}')
        
        csv_path, fig_path = logger.csv_path, logger.fig_path
    
    # Plot learning curve
    plot_curve(csv_path, fig_path, 'cfr')
    print(f"\nTraining complete! Results saved to {args.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR training for NFL")
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_eval_games', type=int, default=200)
    parser.add_argument('--evaluate_every', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='experiments/nfl_cfr/')
    
    args = parser.parse_args()
    train(args)
