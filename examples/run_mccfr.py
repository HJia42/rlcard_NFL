"""
Run MCCFR on NFL Game

Usage:
    python examples/run_mccfr.py --game nfl-bucketed --iterations 1000
    python examples/run_mccfr.py --game nfl --iterations 1000
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rlcard
from rlcard.agents.mccfr_agent import MCCFRAgent
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament


def evaluate_vs_random(env, agent, num_games=500):
    """Evaluate against random agent."""
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    env.set_agents([agent, random_agent])
    result_0 = tournament(env, num_games)
    
    env.set_agents([random_agent, agent])
    result_1 = tournament(env, num_games)
    
    return result_0[0], result_1[1]


def main():
    parser = argparse.ArgumentParser(description='MCCFR on NFL')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed'],
                        help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='./models/mccfr')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print(f"MCCFR on {args.game}")
    print("=" * 60)
    
    # Use full drives for bucketed game, single play for full game
    use_single_play = (args.game == 'nfl')  # Only single play for full game
    
    # Create environment
    env = rlcard.make(args.game, config={
        'seed': args.seed,
        'allow_step_back': True,
        'single_play': use_single_play,
    })
    
    eval_env = rlcard.make(args.game, config={
        'seed': args.seed + 1000,
        'single_play': True,
    })
    
    # Print info set counts (only for bucketed game)
    if args.game == 'nfl-bucketed':
        from rlcard.games.nfl.game_bucketed import NFLGameBucketed
        info_sets = NFLGameBucketed.count_info_sets()
        print(f"\nInfo Set Counts:")
        print(f"  Phase 0 (formation): {info_sets['phase_0']}")
        print(f"  Phase 1 (defense):   {info_sets['phase_1']}")
        print(f"  Phase 2 (play_type): {info_sets['phase_2']}")
        print(f"  Total:               {info_sets['total']}")
    
    # Create MCCFR agent
    agent = MCCFRAgent(env, model_path=args.model_path)
    
    print(f"\nTraining for {args.iterations} iterations...")
    start = time.time()
    
    for i in range(args.iterations):
        agent.train()
        
        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"Iter {i+1:4d} | States: {len(agent.policy):4d} | "
                  f"{(i+1)/elapsed:.1f} iter/sec")
        
        if (i + 1) % args.eval_every == 0:
            print(f"\n--- Evaluation at {i+1} ---")
            off_epa, def_epa = evaluate_vs_random(eval_env, agent)
            print(f"  vs Random (Off): {off_epa:.3f} EPA")
            print(f"  vs Random (Def): {def_epa:.3f} EPA")
            print()
    
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Iterations: {args.iterations}")
    print(f"Time: {elapsed:.1f}s")
    print(f"States discovered: {len(agent.policy)}")
    
    agent.save()
    print(f"Model saved to {args.model_path}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    off_epa, def_epa = evaluate_vs_random(eval_env, agent, 1000)
    print(f"  vs Random (Off): {off_epa:.3f} EPA")
    print(f"  vs Random (Def): {def_epa:.3f} EPA")


if __name__ == '__main__':
    main()
