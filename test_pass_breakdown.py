"""Test pass play outcome breakdown."""
import sys
sys.path.insert(0, '.')

from rlcard.games.nfl.game import NFLGame
import numpy as np

g = NFLGame(use_distribution_model=True)
g.init_game()

n = 500
results = {'completion': [], 'incomplete': 0, 'sack': 0, 'interception': 0, 'td': 0}

for _ in range(n):
    r = g._get_outcome(1, 10, 50, ('SHOTGUN', 'pass'), (6, 'Standard'))
    yds = r['yards_gained']
    if r['turnover']:
        results['interception'] += 1
    elif yds == 0:
        results['incomplete'] += 1
    elif yds < 0:
        results['sack'] += 1
    elif yds >= 50:
        results['td'] += 1
    else:
        results['completion'].append(yds)

n_comp = len(results['completion'])
print("PASS PLAY BREAKDOWN (SHOTGUN vs 6-box, n=500):")
print(f"  Completions: {n_comp}/{n} ({n_comp/n*100:.1f}%)")
if n_comp > 0:
    print(f"    Mean yards on completions: {np.mean(results['completion']):.1f}")
print(f"  Incompletions: {results['incomplete']}/{n} ({results['incomplete']/n*100:.1f}%)")
print(f"  Sacks: {results['sack']}/{n} ({results['sack']/n*100:.1f}%)")
print(f"  Interceptions: {results['interception']}/{n} ({results['interception']/n*100:.1f}%)")
print(f"  TDs: {results['td']}/{n} ({results['td']/n*100:.1f}%)")

# Calculate overall mean including all outcomes
all_yards = results['completion'] + [0]*results['incomplete'] + [-5]*results['sack'] + [50]*results['td']
print(f"\n  OVERALL MEAN (all outcomes): {np.mean(all_yards):.1f} yds")
