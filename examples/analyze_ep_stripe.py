"""Quick script to analyze EP stripe pattern by down."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\jiaha\Projects\NFL_Playcalling\Code\data\cleaned_nfl_rl_data.csv')
df['yardline'] = 100 - df['yardline_100']

# Create scatter by down to see the stripe
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

downs = [1, 2, 3, 4]
for idx, down in enumerate(downs):
    ax = axes[idx // 2, idx % 2]
    subset = df[df['down'] == down]
    ax.scatter(subset['yardline'], subset['ep'], alpha=0.1, s=1)
    ax.set_xlabel('Yardline (from own goal)')
    ax.set_ylabel('EP')
    ax.set_title(f'Down {down} (n={len(subset):,})')
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 7)
    # Add reference line
    ax.plot([0, 100], [0, 7], 'r--', alpha=0.5, label='Linear')

plt.tight_layout()
plt.savefig('examples/ep_by_down_scatter.png', dpi=150)
print('Saved: examples/ep_by_down_scatter.png')

# Check goal_to_go plays
print()
print('=== Goal-to-go analysis ===')
print('Goal-to-go plays:', df['goal_to_go'].sum())
gtg = df[df['goal_to_go'] == 1]
non_gtg = df[df['goal_to_go'] == 0]
print(f'Goal-to-go EP mean: {gtg["ep"].mean():.2f}')
print(f'Non-goal-to-go EP mean: {non_gtg["ep"].mean():.2f}')

# Check if goal_to_go creates the stripe
print()
print('=== Yardline distribution for goal-to-go ===')
print(gtg['yardline'].describe())
