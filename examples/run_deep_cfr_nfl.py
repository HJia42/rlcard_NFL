"""
Run Deep CFR on NFL Game

Deep CFR uses neural networks to approximate regrets, enabling generalization
across similar game states. This is more sample-efficient than tabular MCCFR
for games with large state spaces.

Usage:
    python examples/run_deep_cfr_nfl.py
    python examples/run_deep_cfr_nfl.py --iterations 500 --eval_every 100
    python examples/run_deep_cfr_nfl.py --load  # Resume training
"""
import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlcard
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament


def evaluate_against_random(env, agent, num_games=500):
    """Evaluate Deep CFR agent against random agent."""
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    # Deep CFR as player 0 (offense)
    env.set_agents([agent, random_agent])
    result_as_offense = tournament(env, num_games)
    
    # Deep CFR as player 1 (defense)  
    env.set_agents([random_agent, agent])
    result_as_defense = tournament(env, num_games)
    
    return {
        'offense_payoff': result_as_offense[0],
        'defense_payoff': result_as_defense[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Train Deep CFR on NFL')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Number of training iterations')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N iterations')
    parser.add_argument('--eval_games', type=int, default=500,
                        help='Number of games for evaluation')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer sizes for networks')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--train_steps', type=int, default=100,
                        help='Training steps per iteration')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--model_path', type=str, default='./models/deep_cfr_nfl',
                        help='Path to save/load model')
    parser.add_argument('--load', action='store_true',
                        help='Load existing model and continue training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("Deep CFR Training on NFL Game")
    print("=" * 60)
    
    # Create environment with step_back enabled
    env = rlcard.make(
        'nfl',
        config={
            'seed': args.seed,
            'allow_step_back': True,
            'single_play': True,  # Critical for CFR performance
        }
    )
    
    # Create evaluation environment
    eval_env = rlcard.make(
        'nfl', 
        config={
            'seed': args.seed + 1000,
            'allow_step_back': False,
            'single_play': True,
        }
    )

    print(f"\nEnvironment Info:")
    print(f"  Game: nfl")
    print(f"  Num Actions: {env.num_actions}")
    print(f"  Num Players: {env.num_players}")
    print(f"  State Shape: {env.state_shape}")
    
    # Create Deep CFR agent
    agent = DeepCFRAgent(
        env,
        hidden_layers=args.hidden_layers,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        learning_rate=args.lr,
        model_path=args.model_path
    )
    
    print(f"\nAgent Config:")
    print(f"  Hidden Layers: {args.hidden_layers}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Train Steps: {args.train_steps}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Device: {agent.device}")
    
    # Load existing model if requested
    if args.load:
        if agent.load():
            print(f"\nLoaded model from {args.model_path}")
            print(f"  Resuming from iteration {agent.iteration}")
        else:
            print(f"\nNo model found at {args.model_path}, starting fresh")
    
    print(f"\nTraining for {args.iterations} iterations...")
    print(f"  Evaluating every {args.eval_every} iterations")
    print()
    
    start_time = time.time()
    
    for i in range(args.iterations):
        # Train one iteration
        stats = agent.train()
        
        # Log progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            iter_per_sec = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"Iter {agent.iteration:4d} | "
                  f"Nodes: {stats['nodes_visited']:4d} | "
                  f"Samples: {stats['samples_collected']:3d} | "
                  f"Adv Loss: {stats['advantage_loss']:.4f} | "
                  f"Pol Loss: {stats['policy_loss']:.4f} | "
                  f"{iter_per_sec:.1f} iter/sec")
        
        # Evaluate periodically
        if (i + 1) % args.eval_every == 0:
            print(f"\n--- Evaluation at iteration {agent.iteration} ---")
            eval_results = evaluate_against_random(eval_env, agent, args.eval_games)
            print(f"  vs Random (as Offense): {eval_results['offense_payoff']:.3f} EPA")
            print(f"  vs Random (as Defense): {eval_results['defense_payoff']:.3f} EPA")
            
            # Save checkpoint
            agent.save()
            print(f"  Saved checkpoint to {args.model_path}")
            print()
    
    # Final stats
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total iterations: {agent.iteration}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Advantage buffer sizes: {[len(b) for b in agent.advantage_buffers]}")
    print(f"Policy buffer size: {len(agent.policy_buffer)}")
    
    # Final save
    agent.save()
    print(f"\nModel saved to {args.model_path}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    eval_results = evaluate_against_random(eval_env, agent, args.eval_games * 2)
    print(f"  vs Random (as Offense): {eval_results['offense_payoff']:.3f} EPA")
    print(f"  vs Random (as Defense): {eval_results['defense_payoff']:.3f} EPA")


if __name__ == '__main__':
    main()
