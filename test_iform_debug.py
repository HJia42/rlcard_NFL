"""Investigate I_FORM vs 5-box run yardage."""
import sys
sys.path.insert(0, '.')

from rlcard.games.nfl.game import NFLGame
import numpy as np

g = NFLGame(use_distribution_model=True)

# Check historical data for I_FORM runs vs 5-box
rushes = g.play_data[g.play_data['rush'] == 1]
i_form = rushes[rushes['offense_formation'] == 'I_FORM']
i_form_5box = i_form[i_form['defenders_in_box'] == 5]

print("I_FORM runs vs 5-box:")
print(f"  Total plays: {len(i_form_5box)}")
if len(i_form_5box) > 0:
    gains = i_form_5box['yards_gained'].values
    print(f"  Historical mean: {np.mean(gains):.2f}")
    print(f"  Historical std: {np.std(gains):.2f}")
    print(f"  Historical median: {np.median(gains):.2f}")
    print(f"  Max: {np.max(gains):.0f}")
    
    # Distribution
    print(f"  Plays with 10+ yards: {(gains >= 10).sum()}/{len(gains)} ({(gains >= 10).mean()*100:.1f}%)")
    print(f"  Plays with 20+ yards: {(gains >= 20).sum()}/{len(gains)} ({(gains >= 20).mean()*100:.1f}%)")

print()
print("I_FORM runs vs 6-box:")
i_form_6box = i_form[i_form['defenders_in_box'] == 6]
print(f"  Total plays: {len(i_form_6box)}")
if len(i_form_6box) > 0:
    gains = i_form_6box['yards_gained'].values
    print(f"  Historical mean: {np.mean(gains):.2f}")

print()
print("I_FORM runs vs 7-box:")
i_form_7box = i_form[i_form['defenders_in_box'] == 7]
print(f"  Total plays: {len(i_form_7box)}")
if len(i_form_7box) > 0:
    gains = i_form_7box['yards_gained'].values
    print(f"  Historical mean: {np.mean(gains):.2f}")
