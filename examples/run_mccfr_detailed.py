"""
Run MCCFR (Monte Carlo CFR) on NFL Game

MCCFR uses external sampling to dramatically speed up training compared
to vanilla CFR. Instead of exploring all opponent actions, it samples
according to the current policy.

Usage:
    python examples/run_mccfr_nfl.py
    python examples/run_mccfr_nfl.py --iterations 1000 --eval_every 100
    python examples/run_mccfr_nfl.py --load  # Resume training
"""
import os
import sys
import argparse
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlcard
from rlcard.agents.mccfr_agent import MCCFRAgent
from rlcard.utils.eval_utils import evaluate_agent, format_eval_line, EvalLogger


def main():
    parser = argparse.ArgumentParser(description='Train MCCFR on NFL')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of training iterations')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N iterations')
    parser.add_argument('--eval_games', type=int, default=500,
                        help='Number of games for evaluation')
    parser.add_argument('--model_path', type=str, default='./models/mccfr_nfl',
                        help='Path to save/load model')
    parser.add_argument('--load', action='store_true',
                        help='Load existing model and continue training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    print("=" * 60)
    print("MCCFR Training on NFL Game")
    print("=" * 60)
    
    # Create environment with step_back enabled and single_play mode
    # single_play=True makes game end after one play (tree depth = 3)
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
    
    # Create MCCFR agent
    agent = MCCFRAgent(env, model_path=args.model_path)
    
    # Load existing model if requested
    if args.load:
        agent.load()
        print(f"\nLoaded model from {args.model_path}")
        print(f"  Resuming from iteration {agent.iteration}")
        print(f"  Unique states: {len(agent.regrets)}")
    
    print(f"\nTraining for {args.iterations} iterations...")
    print(f"  Evaluating every {args.eval_every} iterations")
    print()

    os.makedirs(args.model_path, exist_ok=True)
    eval_logger = EvalLogger(os.path.join(args.model_path, 'eval_log.csv'))
    
    start_time = time.time()
    total_nodes = 0
    
    for i in range(args.iterations):
        # Train one iteration
        stats = agent.train()
        total_nodes += stats['nodes_visited']
        
        # Log progress
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            nodes_per_sec = total_nodes / elapsed if elapsed > 0 else 0
            print(f"Iteration {agent.iteration:5d} | "
                  f"Nodes: {stats['nodes_visited']:6d} | "
                  f"Terminal: {stats['terminal_nodes']:4d} | "
                  f"Sampled: {stats['nodes_sampled']:5d} | "
                  f"States: {len(agent.regrets):5d} | "
                  f"{nodes_per_sec:.0f} nodes/sec")
        
        # Evaluate periodically
        if (i + 1) % args.eval_every == 0:
            print(f"\n--- Evaluation at iteration {agent.iteration} ---")
            eval_results = evaluate_agent(agent, game='nfl', num_games=args.eval_games, verbose=False)
            eval_logger.log(agent.iteration, eval_results)
            print(format_eval_line(agent.iteration, eval_results))
            print()
            
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
    print(f"Unique states discovered: {len(agent.regrets)}")
    print(f"Total nodes visited: {total_nodes}")
    print(f"Average nodes/sec: {total_nodes/elapsed:.0f}")
    
    # Final save
    agent.save()
    print(f"\nModel saved to {args.model_path}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    eval_results = evaluate_agent(agent, game='nfl', num_games=args.eval_games * 2, verbose=False)
    print(format_eval_line("Final", eval_results))
    
    # Show sample policies
    print("\n--- Sample Learned Policies ---")
    count = 0
    for obs, probs in agent.average_policy.items():
        if count >= 5:
            print("  ...")
            break
        non_zero = [(j, f"{p:.2f}") for j, p in enumerate(probs) if p > 0.01]
        if non_zero:
            print(f"  State {count}: {non_zero}")
            count += 1


if __name__ == '__main__':
    main()
