"""
Unified Agent Analysis for NFL Play-Calling

Analyze any trained agent (PPO, DMC, NFSP, CFR, MCCFR, Deep CFR) with:
1. 4th Down decision accuracy
2. Defense box count decisions
3. Self-play performance metrics

Usage:
    python examples/analyze_agent.py ppo models/ppo_nfl/ppo_nfl-bucketed_final.pt
    python examples/analyze_agent.py dmc experiments/dmc_result/dmc_nfl-bucketed
    python examples/analyze_agent.py nfsp models/nfsp_nfl/nfsp_nfl-bucketed_p0_final.pt
    python examples/analyze_agent.py cfr models/cfr
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcard.utils.agent_loader import load_agent, SUPPORTED_AGENT_TYPES
from rlcard.utils.analysis_utils import (
    analyze_fourth_down,
    analyze_defense,
    run_self_play_evaluation,
)


def main():
    parser = argparse.ArgumentParser(
        description='Unified NFL Agent Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python examples/analyze_agent.py ppo models/ppo_nfl/ppo_nfl-bucketed_final.pt
  python examples/analyze_agent.py dmc experiments/dmc_result/dmc_nfl-bucketed
  python examples/analyze_agent.py nfsp models/nfsp_nfl/nfsp_nfl-bucketed_p0_final.pt --verbose
  python examples/analyze_agent.py cfr models/cfr --skip-self-play
""")
    
    parser.add_argument('agent_type', type=str, choices=SUPPORTED_AGENT_TYPES,
                       help='Type of agent to analyze')
    parser.add_argument('model_path', type=str,
                       help='Path to model file or directory')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                       choices=['nfl', 'nfl-bucketed', 'nfl-iig', 'nfl-iig-bucketed'],
                       help='Game environment (default: nfl-bucketed)')
    parser.add_argument('--verbose', action='store_true',
                       help='Show all action probabilities')
    parser.add_argument('--skip-self-play', action='store_true',
                       help='Skip self-play evaluation')
    parser.add_argument('--skip-defense', action='store_true',
                       help='Skip defense analysis')
    parser.add_argument('--self-play-games', type=int, default=100,
                       help='Number of self-play games (default: 100)')
    
    args = parser.parse_args()
    
    # Load agent
    print("=" * 70)
    print(f"{args.agent_type.upper()} Agent Analysis")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Game:  {args.game}")
    print()
    
    agent, env = load_agent(args.agent_type, args.model_path, args.game)
    
    if agent is None:
        print("Failed to load agent. Check model path and agent type.")
        sys.exit(1)
    
    # Part 1: 4th Down Analysis
    fourth_correct, fourth_total = analyze_fourth_down(
        agent, env, 
        verbose=args.verbose
    )
    
    # Part 2: Defense Analysis
    if not args.skip_defense:
        defense_results = analyze_defense(
            agent, env,
            verbose=args.verbose
        )
    
    # Part 3: Self-Play Metrics
    if not args.skip_self_play:
        metrics = run_self_play_evaluation(agent, env, args.self_play_games)
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"4th Down Accuracy: {fourth_correct}/{fourth_total} ({100*fourth_correct/fourth_total:.1f}%)")
    
    if not args.skip_self_play:
        print(f"Self-Play Balance:  {metrics['reward_imbalance']:.4f} imbalance")
        print(f"Offense Win Rate:   {metrics['win_rate_0']*100:.1f}%")


if __name__ == '__main__':
    main()
