"""
Unified Agent Convergence Plotting

Plot training convergence from eval_log.csv files.
Supports single agent or multi-agent comparison.

Usage:
    # Single agent
    python examples/plot_agent_convergence.py models/ppo_nfl/eval_log.csv --label PPO
    
    # Multiple agents comparison
    python examples/plot_agent_convergence.py models/ppo_nfl/eval_log.csv models/nfsp_nfl/eval_log.csv --labels PPO NFSP
    
    # Save to file
    python examples/plot_agent_convergence.py models/ppo_nfl/eval_log.csv --save-plot convergence.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlcard.utils.eval_utils import (
    load_eval_history,
    plot_eval_history,
    plot_multi_agent_convergence,
)


def main():
    parser = argparse.ArgumentParser(
        description='Plot agent training convergence',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single agent with all metrics
  python examples/plot_agent_convergence.py models/ppo_nfl/eval_log.csv
  
  # Compare multiple agents
  python examples/plot_agent_convergence.py log1.csv log2.csv --labels PPO NFSP
  
  # Save plot
  python examples/plot_agent_convergence.py models/ppo_nfl/eval_log.csv --save-plot convergence.png
""")
    
    parser.add_argument('csv_paths', type=str, nargs='+',
                       help='Path(s) to eval_log.csv file(s)')
    parser.add_argument('--labels', type=str, nargs='*',
                       help='Labels for each agent (required if multiple CSVs)')
    parser.add_argument('--metric', type=str, default='offense_epa',
                       choices=['offense_epa', 'defense_epa', 'self_play_epa'],
                       help='Metric to plot for comparison (default: offense_epa)')
    parser.add_argument('--title', type=str, default=None,
                       help='Plot title')
    parser.add_argument('--save-plot', type=str, default=None,
                       help='Save plot to file')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display plot (useful with --save-plot)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if len(args.csv_paths) > 1 and not args.labels:
        parser.error("--labels required when comparing multiple agents")
    
    if args.labels and len(args.labels) != len(args.csv_paths):
        parser.error("Number of labels must match number of CSV files")
    
    # Generate default labels if not provided
    if not args.labels:
        args.labels = [os.path.basename(os.path.dirname(p)) or 'Agent' for p in args.csv_paths]
    
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        sys.exit(1)
    
    if len(args.csv_paths) == 1:
        # Single agent: show all three metrics
        print(f"Plotting convergence for {args.labels[0]}...")
        fig = plot_eval_history(
            args.csv_paths[0],
            save_path=args.save_plot,
            title=args.title or f"{args.labels[0]} Training Convergence"
        )
    else:
        # Multiple agents: compare specified metric
        print(f"Comparing {len(args.csv_paths)} agents on {args.metric}...")
        fig = plot_multi_agent_convergence(
            args.csv_paths,
            args.labels,
            save_path=args.save_plot,
            title=args.title,
            metric=args.metric,
        )
    
    if fig and not args.no_show:
        plt.show()


if __name__ == '__main__':
    main()
