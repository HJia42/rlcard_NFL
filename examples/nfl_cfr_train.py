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
from rlcard.utils import set_seed, plot_curve
from rlcard.utils.eval_utils import EvalLogger, evaluate_agent, format_eval_line


def train(args):
    # Handle aliases
    if args.save_dir:
        args.log_dir = args.save_dir
    if args.iterations:
        args.num_episodes = args.iterations

    # Create environments with step_back enabled for CFR
    env_config = {
        'seed': args.seed,
        'allow_step_back': True,
        'single_play': args.single_play,
        'start_down': args.start_down,
    }
    
    env = rlcard.make(args.game, config=env_config)
    
    eval_env = rlcard.make(args.game, config={
        'seed': args.seed,
        'single_play': args.single_play,
        'start_down': args.start_down,
    })
    
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
    
    # Initialize EvalLogger
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, 'eval_log.csv')
    eval_logger = EvalLogger(log_path)
    
    # Training loop
    for episode in range(1, args.num_episodes + 1):
        agent.train()
        print(f'\rIteration {episode}', end='')
        
        if episode % args.evaluate_every == 0:
            agent.save()
            # Evaluate using standardized metrics
            results = evaluate_agent(agent, args.game, num_games=args.num_eval_games, verbose=False)
            results['episode'] = episode  # Add episode to results
            
            # Log and print
            eval_logger.log(episode, results)
            print(f"\n{format_eval_line(episode, results)}")
    
    csv_path = log_path
    fig_path = os.path.join(args.log_dir, 'fig.png')
    
    # Plot standard evaluation history
    from rlcard.utils.eval_utils import plot_eval_history
    plot_eval_history(csv_path, fig_path, title='CFR Training EPA')
    
    print(f"\nTraining complete! Results saved to {args.log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("CFR training for NFL")
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed', 'nfl-iig', 'nfl-iig-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--num_eval_games', type=int, default=200)
    parser.add_argument('--evaluate_every', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='models/cfr/')
    parser.add_argument('--save-dir', type=str, default=None, help='Alias for log_dir')
    parser.add_argument('--iterations', type=int, default=None, help='Alias for num_episodes')
    
    # Custom game config
    parser.add_argument('--single-play', action='store_true', help='End game after one play')
    parser.add_argument('--start-down', type=int, default=1, help='Starting down (1-4)')
    
    args = parser.parse_args()
    train(args)
