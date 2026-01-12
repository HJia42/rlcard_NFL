"""Test run play outcome breakdown."""
import sys
sys.path.insert(0, '.')

from rlcard.games.nfl.game import NFLGame
import numpy as np

g = NFLGame(use_distribution_model=True)
g.init_game()

n = 500

# Test run play from I_FORM
results = {'positive': [], 'zero': 0, 'loss': 0, 'fumble': 0, 'td': 0}

for _ in range(n):
    r = g._get_outcome(1, 10, 50, ('I_FORM', 'rush'), (6, 'Standard'))
    yds = r['yards_gained']
    if r['turnover']:
        results['fumble'] += 1
    elif yds >= 50:
        results['td'] += 1
    elif yds > 0:
        results['positive'].append(yds)
    elif yds == 0:
        results['zero'] += 1
    else:
        results['loss'] += 1

n_pos = len(results['positive'])
print("RUN PLAY BREAKDOWN (I_FORM vs 6-box, n=500):")
print(f"  Positive gains: {n_pos}/{n} ({n_pos/n*100:.1f}%)")
if n_pos > 0:
    print(f"    Mean yards on gains: {np.mean(results['positive']):.1f}")
print(f"  No gain (0 yds): {results['zero']}/{n} ({results['zero']/n*100:.1f}%)")
print(f"  Losses (<0 yds): {results['loss']}/{n} ({results['loss']/n*100:.1f}%)")
print(f"  Fumbles: {results['fumble']}/{n} ({results['fumble']/n*100:.1f}%)")
print(f"  TDs: {results['td']}/{n} ({results['td']/n*100:.1f}%)")

all_yards = results['positive'] + [0]*results['zero'] + [-3]*results['loss'] + [50]*results['td']
print(f"\n  OVERALL MEAN: {np.mean(all_yards):.1f} yds")

# Compare to actual data stats
print("\n" + "="*60)
print("COMPARISON TO HISTORICAL DATA:")
print("="*60)
rushes = g.play_data[g.play_data['rush'] == 1]
i_form = rushes[rushes['offense_formation'] == 'I_FORM']
print(f"Total I_FORM rushes in data: {len(i_form)}")
print(f"Historical mean: {i_form['yards_gained'].mean():.2f} yds")
print(f"Historical std: {i_form['yards_gained'].std():.2f}")
print(f"Historical median: {i_form['yards_gained'].median():.2f}")
print(f"Positive gains: {(i_form['yards_gained'] > 0).sum()}/{len(i_form)} ({(i_form['yards_gained'] > 0).mean()*100:.1f}%)")
print(f"Zero gains: {(i_form['yards_gained'] == 0).sum()}/{len(i_form)} ({(i_form['yards_gained'] == 0).mean()*100:.1f}%)")
print(f"Losses: {(i_form['yards_gained'] < 0).sum()}/{len(i_form)} ({(i_form['yards_gained'] < 0).mean()*100:.1f}%)")
