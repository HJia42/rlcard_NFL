"""
Training Convergence Analysis for NFL RL Agents

Evaluates agent checkpoints across training to plot learning curves.
Produces data and plots for paper publication.

Usage:
    python examples/plot_convergence.py models_cloud/ --pattern "ppo_*.pt"
    python examples/plot_convergence.py models_cloud/ --pattern "ppo_*.pt" --save-plot convergence.png
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import rlcard
from rlcard.agents.ppo_agent import PPOAgent


# Test scenarios for 4th down accuracy
FOURTH_DOWN_TESTS = [
    (10, 10, 4, "PUNT"),   # 4th & 10 at own 10
    (25, 10, 4, "PUNT"),   # 4th & 10 at own 25
    (75, 10, 4, "FG"),     # 4th & 10 at opp 25
    (85, 8, 4, "FG"),      # 4th & 8 at opp 15
    (95, 3, 4, "GO"),      # 4th & Goal at 5
    (99, 1, 4, "GO"),      # 4th & Goal at 1
]

OFFENSE_ACTIONS = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY', 'PUNT', 'FG']


def extract_iteration(filename):
    """Extract iteration number from checkpoint filename."""
    match = re.search(r'_(\d+)\.pt$', filename)
    if match:
        return int(match.group(1))
    if 'final' in filename.lower():
        return float('inf')  # Final model goes last
    return 0


def load_agent(model_path, env):
    """Load a PPO agent from checkpoint."""
    agent = PPOAgent(
        state_shape=env.state_shape[0], 
        num_actions=7, 
        hidden_dims=[128, 128]
    )
    agent.load(model_path)
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
        action, probs = agent.eval_step(state)
        
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        top_action = OFFENSE_ACTIONS[sorted_probs[0][0]]
        
        if expected == "GO":
            is_correct = top_action not in ['PUNT', 'FG']
        else:
            is_correct = (top_action == expected)
        
        if is_correct:
            correct += 1
    
    return correct / len(FOURTH_DOWN_TESTS)


def evaluate_self_play(agent, env, num_games=50):
    """Run self-play games and return average reward."""
    total_reward_0 = 0
    total_reward_1 = 0
    
    for _ in range(num_games):
        state, player_id = env.reset()
        
        while not env.is_over():
            action, _ = agent.eval_step(state)
            state, player_id = env.step(action)
        
        payoffs = env.get_payoffs()
        total_reward_0 += payoffs[0]
        total_reward_1 += payoffs[1]
    
    return {
        'avg_reward_offense': total_reward_0 / num_games,
        'avg_reward_defense': total_reward_1 / num_games,
        'reward_gap': abs(total_reward_0 - total_reward_1) / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Plot training convergence')
    parser.add_argument('model_dir', type=str, 
                        help='Directory containing model checkpoints')
    parser.add_argument('--pattern', type=str, default='ppo_*.pt',
                        help='Glob pattern for checkpoint files')
    parser.add_argument('--self-play-games', type=int, default=30,
                        help='Number of self-play games per checkpoint')
    parser.add_argument('--save-plot', type=str, default=None,
                        help='Save plot to file')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Save data to CSV file')
    args = parser.parse_args()
    
    # Find checkpoint files
    pattern = os.path.join(args.model_dir, args.pattern)
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print(f"No files found matching: {pattern}")
        return
    
    # Sort by iteration
    checkpoint_files = sorted(checkpoint_files, key=extract_iteration)
    
    # Filter out 'final' for cleaner plots
    checkpoint_files = [f for f in checkpoint_files if 'final' not in f.lower()]
    
    print(f"Found {len(checkpoint_files)} checkpoints")
    
    # Create environment
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True, 
        'use_cached_model': True
    })
    
    # Evaluate each checkpoint
    results = {
        'iteration': [],
        'fourth_down_accuracy': [],
        'avg_reward_offense': [],
        'avg_reward_defense': [],
        'reward_gap': [],
    }
    
    for i, checkpoint in enumerate(checkpoint_files):
        iteration = extract_iteration(checkpoint)
        if iteration == float('inf'):
            continue
            
        print(f"[{i+1}/{len(checkpoint_files)}] Evaluating iteration {iteration}...")
        
        agent = load_agent(checkpoint, env)
        
        # 4th down accuracy
        accuracy = evaluate_fourth_down_accuracy(agent, env)
        
        # Self-play metrics
        sp_metrics = evaluate_self_play(agent, env, args.self_play_games)
        
        results['iteration'].append(iteration)
        results['fourth_down_accuracy'].append(accuracy)
        results['avg_reward_offense'].append(sp_metrics['avg_reward_offense'])
        results['avg_reward_defense'].append(sp_metrics['avg_reward_defense'])
        results['reward_gap'].append(sp_metrics['reward_gap'])
        
        print(f"  4th Down: {accuracy*100:.1f}%, EPA Offense: {sp_metrics['avg_reward_offense']:.3f}")
    
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Plot 1: 4th Down Accuracy
    ax1 = axes[0, 0]
    ax1.plot(results['iteration'], [a*100 for a in results['fourth_down_accuracy']], 
             'b-o', linewidth=2, markersize=4)
    ax1.set_xlabel('Training Episodes')
    ax1.set_ylabel('4th Down Accuracy (%)')
    ax1.set_title('4th Down Decision Accuracy')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 100)
    
    # Plot 2: Average Reward (EPA)
    ax2 = axes[0, 1]
    ax2.plot(results['iteration'], results['avg_reward_offense'], 
             'g-o', label='Offense', linewidth=2, markersize=4)
    ax2.plot(results['iteration'], results['avg_reward_defense'], 
             'r-o', label='Defense', linewidth=2, markersize=4)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Training Episodes')
    ax2.set_ylabel('Average EPA per Game')
    ax2.set_title('Self-Play EPA Convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Reward Gap (Exploitability Proxy)
    ax3 = axes[1, 0]
    ax3.plot(results['iteration'], results['reward_gap'], 
             'm-o', linewidth=2, markersize=4)
    ax3.set_xlabel('Training Episodes')
    ax3.set_ylabel('|Offense EPA - Defense EPA|')
    ax3.set_title('Strategy Imbalance (Lower = More Balanced)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Stats as Table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary table
    final_idx = -1
    summary_text = f"""
    TRAINING CONVERGENCE SUMMARY
    ============================
    
    Total Checkpoints: {len(results['iteration'])}
    Training Episodes: {min(results['iteration']):,} → {max(results['iteration']):,}
    
    4th Down Accuracy:
      Start: {results['fourth_down_accuracy'][0]*100:.1f}%
      Final: {results['fourth_down_accuracy'][final_idx]*100:.1f}%
    
    Offense EPA:
      Start: {results['avg_reward_offense'][0]:.4f}
      Final: {results['avg_reward_offense'][final_idx]:.4f}
    
    Strategy Balance:
      Start: {results['reward_gap'][0]:.4f}
      Final: {results['reward_gap'][final_idx]:.4f}
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    if args.save_plot:
        plt.savefig(args.save_plot, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.save_plot}")
    
    plt.show()


if __name__ == '__main__':
    main()
