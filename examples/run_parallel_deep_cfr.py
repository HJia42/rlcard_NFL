"""
Run Parallel Deep CFR on NFL Game

Uses multiple actor processes to collect samples in parallel while
training networks on GPU. Significantly faster than sequential training.

Usage:
    python examples/run_parallel_deep_cfr.py --num_actors 4 --cuda
    python examples/run_parallel_deep_cfr.py --iterations 500 --eval_every 100
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlcard
from rlcard.agents.parallel_deep_cfr import ParallelDeepCFRTrainer
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament


def evaluate_against_random(env, agent, num_games=500):
    """Evaluate agent against random."""
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    env.set_agents([agent, random_agent])
    result_offense = tournament(env, num_games)
    
    env.set_agents([random_agent, agent])
    result_defense = tournament(env, num_games)
    
    return {
        'offense': result_offense[0],
        'defense': result_defense[1],
    }


def main():
    parser = argparse.ArgumentParser(description='Parallel Deep CFR on NFL')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Training iterations')
    parser.add_argument('--eval_every', type=int, default=100,
                        help='Evaluate every N iterations')
    parser.add_argument('--num_actors', type=int, default=4,
                        help='Number of parallel actor processes')
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[128, 128],
                        help='Network hidden layers')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--cuda', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--model_path', type=str, default='./models/parallel_deep_cfr',
                        help='Model save path')
    parser.add_argument('--full_game', action='store_true',
                        help='Use full-game mode (entire drive, slower)')
    parser.add_argument('--seed', type=int, default=42)
    # Custom starting state parameters
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
    args = parser.parse_args()
    
    print("=" * 60)
    print("Parallel Deep CFR Training on NFL")
    print("=" * 60)
    
    env_config = {
        'seed': args.seed,
        'allow_step_back': True,
        'single_play': not args.full_game,
        'start_down': args.start_down,
        'start_ydstogo': args.start_ydstogo,
        'start_yardline': args.start_yardline,
        'use_distribution_model': args.distribution_model,
        'use_cached_model': args.cached_model,
    }
    
    print(f"\nConfig:")
    print(f"  Actors: {args.num_actors}")
    print(f"  GPU: {args.cuda}")
    print(f"  Hidden Layers: {args.hidden_layers}")
    print(f"  Single Play: {not args.full_game}")
    if args.start_down:
        print(f"  Custom Start: {args.start_down} & {args.start_ydstogo or 10} at {args.start_yardline or 25}")
    
    # Create trainer
    trainer = ParallelDeepCFRTrainer(
        env_config=env_config,
        num_actors=args.num_actors,
        hidden_layers=args.hidden_layers,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        cuda=args.cuda,
        model_path=args.model_path,
    )
    
    print(f"  Device: {trainer.device}")
    
    # Create eval env (with same config as training)
    eval_env = rlcard.make(args.game, config={
        'seed': args.seed + 1000,
        'single_play': not args.full_game,
        'use_distribution_model': args.distribution_model,
        'use_cached_model': args.cached_model,
    })
    
    print(f"\nStarting {args.num_actors} actor processes...")
    trainer.start_actors()
    
    # Give actors time to start
    time.sleep(2)
    
    print(f"\nTraining for {args.iterations} iterations...")
    start_time = time.time()
    
    try:
        for i in range(args.iterations):
            stats = trainer.train_step()
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Iter {trainer.iteration:4d} | "
                      f"Samples: {stats['samples']:4d} | "
                      f"Adv: {stats['adv_loss']:.4f} | "
                      f"Pol: {stats['pol_loss']:.4f} | "
                      f"Buf: {stats['adv_buffer']} | "
                      f"{(i+1)/elapsed:.1f} iter/sec")
            
            if (i + 1) % args.eval_every == 0:
                print(f"\n--- Evaluation at iter {trainer.iteration} ---")
                agent = trainer.get_agent()
                results = evaluate_against_random(eval_env, agent, 500)
                print(f"  vs Random (Off): {results['offense']:.3f} EPA")
                print(f"  vs Random (Def): {results['defense']:.3f} EPA")
                trainer.save()
                print(f"  Saved to {args.model_path}")
                print()
    
    finally:
        print("\nStopping actors...")
        trainer.stop_actors()
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Iterations: {trainer.iteration}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Throughput: {trainer.iteration/elapsed:.1f} iter/sec")
    
    trainer.save()
    print(f"\nModel saved to {args.model_path}")


if __name__ == '__main__':
    main()
