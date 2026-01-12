"""Test the outcome model to verify sensible results."""
import sys
sys.path.insert(0, '.')

from rlcard.games.nfl.game import NFLGame
import numpy as np

def test_scenarios():
    g = NFLGame(use_distribution_model=True)
    g.init_game()
    
    def test_scenario(formation, play_type, box, n=300):
        yards_list = []
        turnovers = 0
        for _ in range(n):
            result = g._get_outcome(1, 10, 50, (formation, play_type), (box, 'Standard'))
            yards_list.append(result['yards_gained'])
            if result['turnover']:
                turnovers += 1
        return np.mean(yards_list), np.std(yards_list), turnovers/n
    
    print("=" * 70)
    print("RUN PLAYS (1st & 10 at midfield, 300 samples each)")
    print("=" * 70)
    print(f"{'Formation':<15} {'Box':<5} {'Mean Yds':>10} {'Std':>8} {'TO Rate':>10}")
    print("-" * 70)
    
    formations = ['I_FORM', 'SHOTGUN', 'SINGLEBACK', 'UNDER CENTER']
    for form in formations:
        for box in [5, 6, 7, 8]:
            mean, std, to_rate = test_scenario(form, 'rush', box)
            print(f"{form:<15} {box:<5} {mean:>10.1f} {std:>8.1f} {to_rate*100:>9.1f}%")
    
    print()
    print("=" * 70)
    print("PASS PLAYS (1st & 10 at midfield, 300 samples each)")
    print("=" * 70)
    print(f"{'Formation':<15} {'Box':<5} {'Mean Yds':>10} {'Std':>8} {'TO Rate':>10}")
    print("-" * 70)
    
    for form in formations:
        for box in [5, 6, 7]:
            mean, std, to_rate = test_scenario(form, 'pass', box)
            print(f"{form:<15} {box:<5} {mean:>10.1f} {std:>8.1f} {to_rate*100:>9.1f}%")

if __name__ == '__main__':
    test_scenarios()
