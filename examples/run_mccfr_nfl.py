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
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament


def evaluate_against_random(env, agent, num_games=1000):
    """Evaluate MCCFR agent against random agent.
    
    Args:
        env: RLCard environment
        agent: Trained MCCFR agent
        num_games: Number of evaluation games
        
    Returns:
        Dict with win rates for each player
    """
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    # MCCFR as player 0 (offense)
    env.set_agents([agent, random_agent])
    result_as_offense = tournament(env, num_games)
    
    # MCCFR as player 1 (defense)  
    env.set_agents([random_agent, agent])
    result_as_defense = tournament(env, num_games)
    
    return {
        'offense_payoff': result_as_offense[0],
        'defense_payoff': result_as_defense[1],
    }


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
    
    # Create environment with step_back enabled
    env = rlcard.make(
        'nfl',
        config={
            'seed': args.seed,
            'allow_step_back': True,
        }
    )
    
    # Create evaluation environment (no step_back needed)
    eval_env = rlcard.make(
        'nfl', 
        config={
            'seed': args.seed + 1000,
            'allow_step_back': False,
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
            eval_results = evaluate_against_random(eval_env, agent, args.eval_games)
            print(f"  vs Random (as Offense): {eval_results['offense_payoff']:.3f} EPA")
            print(f"  vs Random (as Defense): {eval_results['defense_payoff']:.3f} EPA")
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
    eval_results = evaluate_against_random(eval_env, agent, args.eval_games * 2)
    print(f"  vs Random (as Offense): {eval_results['offense_payoff']:.3f} EPA")
    print(f"  vs Random (as Defense): {eval_results['defense_payoff']:.3f} EPA")
    
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
