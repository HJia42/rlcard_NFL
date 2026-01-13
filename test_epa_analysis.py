"""EPA Analysis: 4th & 10 at own 25"""
import sys
sys.path.insert(0, '.')
from rlcard.games.nfl.game import NFLGame
import numpy as np

g = NFLGame(use_distribution_model=True)
g.init_game()

print('EPA Analysis: 4th & 10 at own 25')
print('='*60)

# Starting EPA at 4th & 10 at own 25
start_ep = g._calculate_ep(4, 10, 25, False)
print(f'Starting EP at 4th & 10 at own 25: {start_ep:.2f}')

# Option 1: PUNT
print('\n--- OPTION 1: PUNT ---')
punt_epas = []
for _ in range(500):
    g.yardline = 25
    opp_start = g.special_teams.predict_punt_outcome(g.yardline)
    # opp_start is opponent's yardline from THEIR goal
    # Opponent's EP at their yardline (from their goal)
    opp_ep = g._calculate_ep(1, 10, opp_start, False)
    punt_epa = -opp_ep - start_ep  # Negative because opponent gets ball
    punt_epas.append(punt_epa)
print(f'Opponent starts at their: {g.special_teams.predict_punt_outcome(25):.1f}')
print(f'PUNT Expected EPA: {np.mean(punt_epas):.3f}')

# Option 2: GO FOR IT (Pass from I_FORM)
print('\n--- OPTION 2: GO FOR IT (I_FORM pass) ---')
go_epas = []
success_count = 0
for _ in range(500):
    result = g._get_outcome(4, 10, 25, ('I_FORM', 'pass'), (6, 'Standard'))
    yards = result['yards_gained']
    turnover = result['turnover']
    
    if turnover:
        opp_ep = g._calculate_ep(1, 10, 75, False)
        epa = -opp_ep - start_ep
    elif yards >= 10:
        new_yardline = min(99, 25 + yards)
        success_count += 1
        if new_yardline >= 100:
            epa = 7 - start_ep
        else:
            new_ep = g._calculate_ep(1, 10, new_yardline, False)
            epa = new_ep - start_ep
    else:
        opp_start = 100 - (25 + max(0, yards))
        opp_ep = g._calculate_ep(1, 10, opp_start, False)
        epa = -opp_ep - start_ep
    
    go_epas.append(epa)

print(f'Conversion rate (10+ yds): {success_count/500*100:.1f}%')
print(f'GO FOR IT Expected EPA: {np.mean(go_epas):.3f}')

# Summary
print('\n' + '='*60)
print('SUMMARY:')
print(f'  PUNT EPA:       {np.mean(punt_epas):+.3f}')
print(f'  GO FOR IT EPA:  {np.mean(go_epas):+.3f}')
if np.mean(punt_epas) > np.mean(go_epas):
    print('  Better option:  PUNT')
else:
    print('  Better option:  GO FOR IT')
