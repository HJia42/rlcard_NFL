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
from rlcard.utils.eval_utils import evaluate_agent, format_eval_line, EvalLogger


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
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device for training (auto: use GPU if available)')
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

    os.makedirs(args.model_path, exist_ok=True)
    eval_logger = EvalLogger(os.path.join(args.model_path, 'eval_log.csv'))
    
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
            eval_results = evaluate_agent(agent, game='nfl', num_games=args.eval_games, verbose=False)
            eval_logger.log(agent.iteration, eval_results)
            print(format_eval_line(agent.iteration, eval_results))
            
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
    eval_results = evaluate_agent(agent, game='nfl', num_games=args.eval_games * 2, verbose=False)
    print(format_eval_line("Final", eval_results))


if __name__ == '__main__':
    main()
