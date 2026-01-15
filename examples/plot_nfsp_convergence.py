"""
NFSP Training Convergence Analysis

Evaluates NFSP checkpoints across training to plot learning curves.
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt

import torch
import rlcard
from rlcard.agents.nfsp_agent import NFSPAgent


OFFENSE_ACTIONS = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY', 'PUNT', 'FG']

# Test scenarios: (yardline, ydstogo, down, expected)
FOURTH_DOWN_TESTS = [
    (10, 10, 4, "PUNT"),   # 4th & 10 at own 10
    (25, 10, 4, "PUNT"),   # 4th & 10 at own 25
    (75, 10, 4, "FG"),     # 4th & 10 at opp 25
    (85, 8, 4, "FG"),      # 4th & 8 at opp 15
    (95, 3, 4, "GO"),      # 4th & Goal at 5
    (99, 1, 4, "GO"),      # 4th & Goal at 1
]


def extract_iteration(filename):
    """Extract iteration number from checkpoint filename."""
    match = re.search(r'_(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))
    if 'final' in filename.lower():
        return float('inf')
    return 0


def load_nfsp_agent(checkpoint_path, env):
    """Load NFSP agent from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent = NFSPAgent.from_checkpoint(checkpoint)
    return agent


def evaluate_fourth_down_accuracy(agent, env):
    """Evaluate 4th down decision accuracy."""
    correct = 0
    for yardline, ydstogo, down, expected in FOURTH_DOWN_TESTS:
        env.reset()
        env.game.down = down
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        
        state = env.get_state(0)
        action, info = agent.eval_step(state)
        probs = info.get('probs', {})
        
        # Get top action from probs
        if probs:
            top_action = max(probs, key=probs.get)
        else:
            top_action = OFFENSE_ACTIONS[action]
        
        if expected == "GO":
            is_correct = top_action not in ['PUNT', 'FG']
        else:
            is_correct = (top_action == expected)
        
        if is_correct:
            correct += 1
    
    return correct / len(FOURTH_DOWN_TESTS)


def main():
    parser = argparse.ArgumentParser(description='Plot NFSP training convergence')
    parser.add_argument('model_dir', type=str, 
                        help='Directory containing NFSP checkpoints')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save plot to file')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Save data to CSV file')
    args = parser.parse_args()
    
    # Find checkpoint files for player 0
    pattern = os.path.join(args.model_dir, 'nfsp_*_p0_*.pt')
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print(f"No files found matching: {pattern}")
        return
    
    # Sort by iteration and filter out final
    checkpoint_files = [f for f in checkpoint_files if 'final' not in f.lower()]
    checkpoint_files = sorted(checkpoint_files, key=extract_iteration)
    
    print(f"Found {len(checkpoint_files)} NFSP checkpoints")
    
    # Create environment
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True, 
        'use_cached_model': True
    })
    
    # Evaluate each checkpoint
    results = {
        'iteration': [],
        'fourth_down_accuracy': [],
    }
    
    for i, checkpoint in enumerate(checkpoint_files):
        iteration = extract_iteration(checkpoint)
        if iteration == float('inf') or iteration == 0:
            continue
            
        print(f"[{i+1}/{len(checkpoint_files)}] Evaluating iteration {iteration}...")
        
        try:
            agent = load_nfsp_agent(checkpoint, env)
            accuracy = evaluate_fourth_down_accuracy(agent, env)
            
            results['iteration'].append(iteration)
            results['fourth_down_accuracy'].append(accuracy)
            
            print(f"  4th Down: {accuracy*100:.1f}%")
        except Exception as e:
            print(f"  Error: {e}")
    
    # Save to CSV if requested
    if args.save_csv:
        import csv
        with open(args.save_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(results.keys())
            for i in range(len(results['iteration'])):
                writer.writerow([results[k][i] for k in results.keys()])
        print(f"Saved data to {args.save_csv}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(results['iteration'], [a*100 for a in results['fourth_down_accuracy']], 
            'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Training Episodes')
    ax.set_ylabel('4th Down Accuracy (%)')
    ax.set_title('NFSP Training Convergence - 4th Down Decision Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add horizontal line for random baseline
    ax.axhline(y=100/7*2, color='r', linestyle='--', alpha=0.5, label='Random baseline')
    ax.legend()
    
    plt.tight_layout()
    
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.save_plot}")
    
    plt.show()
    
    # Summary
    print("\n" + "="*50)
    print("NFSP TRAINING SUMMARY")
    print("="*50)
    print(f"Checkpoints evaluated: {len(results['iteration'])}")
    print(f"Start accuracy: {results['fourth_down_accuracy'][0]*100:.1f}%")
    print(f"Final accuracy: {results['fourth_down_accuracy'][-1]*100:.1f}%")


if __name__ == '__main__':
    main()
