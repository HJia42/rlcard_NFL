"""Train NFSP agents on NFL game

Usage:
    python examples/nfl_nfsp_train.py --game nfl-bucketed --episodes 10000
"""

import sys
sys.path.insert(0, '.')

import os
import argparse
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import reorganize
from rlcard.utils.eval_utils import quick_eval, format_eval_line


def train(args):
    # Determine single_play mode
    if args.single_play:
        use_single_play = True
    elif args.full_drive:
        use_single_play = False
    else:
        use_single_play = (args.game == 'nfl')  # Default: bucketed=full drive
    
    # Create environment
    env_config = {
        'seed': args.seed,
        'single_play': use_single_play,
        'start_down': args.start_down,
        'start_ydstogo': args.start_ydstogo,
        'start_yardline': args.start_yardline,
        'use_distribution_model': args.distribution_model,
        'use_cached_model': args.cached_model,
    }
    env = rlcard.make(args.game, config=env_config)
    eval_env = rlcard.make(args.game, config={
        'seed': args.seed + 1,
        'single_play': True,
        'use_distribution_model': args.distribution_model,
        'use_cached_model': args.cached_model,
    })
    
    print(f"NFSP Training on {args.game}")
    print(f"  Players: {env.num_players}, Actions: {env.num_actions}")
    
    # Create NFSP agents for both offense and defense
    agents = []
    for i in range(env.num_players):
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=[11],  # Fixed state size (padded)
            hidden_layers_sizes=args.hidden,
            q_mlp_layers=args.hidden,
            anticipatory_param=args.anticipatory_param,
            rl_learning_rate=args.rl_lr,
            sl_learning_rate=args.sl_lr,
            epsilon_start=args.epsilon_start,
            epsilon_end=args.epsilon_end,
        )
        agents.append(agent)
    
    env.set_agents(agents)
    eval_env.set_agents(agents)
    
    print(f"\nTraining NFSP for {args.episodes} episodes...")
    
    for ep in range(1, args.episodes + 1):
        # Train one episode
        trajectories, payoffs = env.run(is_training=True)
        
        # Reorganize trajectories into proper format
        trajectories = reorganize(trajectories, payoffs)
        
        # Feed transitions to agents
        for i in range(env.num_players):
            for transition in trajectories[i]:
                agents[i].feed(transition)
        
        # Evaluate periodically using standardized evaluation
        if ep % args.eval_every == 0:
            # Use first agent for evaluation (both are equivalent in self-play)
            results = quick_eval(agents[0], args.game, num_games=100)
            print(format_eval_line(ep, results))
        
        # Save periodically
        if ep % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                agent.save_checkpoint(args.save_dir, filename=f'nfsp_{args.game}_p{i}_{ep}.pt')
            print(f"Saved models at episode {ep}")
    
    print("\nTraining complete!")
    
    # Final save
    os.makedirs(args.save_dir, exist_ok=True)
    for i, agent in enumerate(agents):
        agent.save_checkpoint(args.save_dir, filename=f'nfsp_{args.game}_p{i}_final.pt')
    print(f"Final models saved to {args.save_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NFSP Training for NFL')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--episodes', type=int, default=10000,
                        help='Number of training episodes')
    parser.add_argument('--eval-every', type=int, default=500,
                        help='Evaluate every N episodes')
    parser.add_argument('--save-every', type=int, default=2000,
                        help='Save every N episodes')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--save-dir', type=str, default='models/nfsp_nfl',
                        help='Save directory')
    parser.add_argument('--seed', type=int, default=42)
    # Single-play and starting state parameters
    parser.add_argument('--single-play', action='store_true',
                        help='End game after one play')
    parser.add_argument('--full-drive', action='store_true',
                        help='Run full drives (default for bucketed)')
    parser.add_argument('--start-down', type=int, default=None, choices=[1, 2, 3, 4],
                        help='Starting down (1-4)')
    parser.add_argument('--start-ydstogo', type=int, default=None,
                        help='Starting yards to go')
    parser.add_argument('--start-yardline', type=int, default=None,
                        help='Starting yardline (1-99, from own goal)')
    parser.add_argument('--distribution-model', action='store_true',
                        help='Use Biro & Walker distribution model for outcomes')
    parser.add_argument('--cached-model', action='store_true',
                        help='Use cached distribution model (O(1) lookup, faster)')
    parser.add_argument('--anticipatory-param', type=float, default=0.1,
                        help='Exploration parameter (0.01-0.3, lower=more exploitation)')
    parser.add_argument('--rl-lr', type=float, default=0.0001,
                        help='RL (DQN) learning rate')
    parser.add_argument('--sl-lr', type=float, default=0.0005,
                        help='Supervised learning rate for policy network')
    parser.add_argument('--epsilon-start', type=float, default=0.06,
                        help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.0,
                        help='Ending epsilon for exploration')
    
    args = parser.parse_args()
    
    if args.start_down:
        print(f"Custom start: {args.start_down} & {args.start_ydstogo or 10} at {args.start_yardline or 25}")
    
    train(args)
