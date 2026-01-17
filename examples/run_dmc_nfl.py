"""
Deep Monte Carlo (DMC) Training for NFL

Trains DMC agents via self-play with proper termination and standardized evaluation.

Usage:
    python examples/run_dmc_nfl.py --game nfl-bucketed --iterations 50000
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import rlcard
from rlcard.agents.dmc_agent import DMCTrainer
from rlcard.utils.eval_utils import evaluate_agent, format_eval_line, EvalLogger


def train_dmc(args):
    """Train DMC agent with standardized output."""
    
    print("=" * 60)
    print("DMC Self-Play Training for NFL")
    print("=" * 60)
    print(f"Game: {args.game}")
    print(f"Iterations: {args.iterations:,}")
    print(f"Save directory: {args.save_dir}")
    print(f"GPU: {args.cuda if args.cuda else 'CPU'}")
    print("=" * 60 + "\n")
    
    # Set CUDA
    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    # Create environment
    # Create environment
    env_config = {
        'single_play': args.single_play,
        'start_down': args.start_down,
        'use_distribution_model': args.distribution_model,
        'use_cached_model': args.cached_model,
    }
    env = rlcard.make(args.game, config=env_config)
    
    # Create eval logger
    log_path = os.path.join(args.save_dir, f'dmc_{args.game}', 'eval_log.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    eval_logger = EvalLogger(log_path)
    
    # Define evaluation callback
    def eval_callback(agent, frames):
        try:
            results = evaluate_agent(agent, args.game, num_games=200, verbose=False) # Keep games low for speed
            results['episode'] = frames # Use frames as episode counter
            eval_logger.log(frames, results)
            print(f"\n[Eval @ {frames} frames] {format_eval_line(frames, results)}")
        except Exception as e:
            print(f"Evaluation failed: {e}")

    # Create DMC trainer
    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        load_model=args.load_model,
        xpid=f'dmc_{args.game}',
        savedir=args.save_dir,
        save_interval=args.save_interval,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device,
        total_frames=args.iterations,  # Total training steps
        eval_every=args.eval_every,
        eval_callback=eval_callback,
    )
    
    print("Starting DMC training...")
    print("(Training will automatically terminate after specified iterations)\n")
    
    start_time = time.time()
    
    try:
        # Start training (this runs in a loop internally)
        trainer.start()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("DMC Training Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Models saved to: {args.save_dir}/dmc_{args.game}/")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    try:
        # Get the trained agent from trainer
        agent = trainer.get_agent()
        results = evaluate_agent(agent, args.game, num_games=500)
        print(format_eval_line("Final", results))
    except Exception as e:
        print(f"Could not evaluate: {e}")
    
    # Clean exit - kill any remaining actor processes
    print("\nCleaning up processes...")
    os._exit(0)  # Force exit to clean up multiprocessing actors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DMC Training for NFL')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed', 'nfl-iig', 'nfl-iig-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='Total training iterations/frames')
    parser.add_argument('--cuda', type=str, default='',
                        help='GPU device ID (empty for CPU)')
    parser.add_argument('--load-model', action='store_true',
                        help='Load existing model to continue training')
    parser.add_argument('--save-dir', type=str, default='models/dmc',
                        help='Save directory')
    parser.add_argument('--save-interval', type=int, default=30,
                        help='Save interval in minutes')
    parser.add_argument('--eval-every', type=int, default=10000,
                        help='Evaluate every N frames/steps')
    parser.add_argument('--num-actor-devices', type=int, default=1,
                        help='Number of actor devices')
    parser.add_argument('--num-actors', type=int, default=5,
                        help='Number of actors per device')
    parser.add_argument('--training-device', type=str, default='0',
                        help='GPU device for training')
    parser.add_argument('--distribution-model', action='store_true',
                        help='Use Biro & Walker distribution model for outcomes')
    parser.add_argument('--cached-model', action='store_true',
                        help='Use cached distribution model (O(1) lookup, faster)')
    parser.add_argument('--single-play', action='store_true',
                        help='End game after one play')
    parser.add_argument('--start-down', type=int, default=1,
                        help='Starting down (1-4)')
    
    args = parser.parse_args()
    train_dmc(args)

