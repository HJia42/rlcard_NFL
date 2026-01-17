import numpy as np
import rlcard
from rlcard.games.nfl.game import NFLGame

def verify_discrete():
    game = NFLGame(single_play=False)
    game.init_game()
    game.yardline = 25
    game.ydstogo = 10
    
    print(f"Start: Yardline={game.yardline}, Type={type(game.yardline)}")
    
    # Simulate a fake outcome
    # Phase 0: Formation
    game.step('SHOTGUN')
    # Phase 1: Defense
    game.step((6, 'Standard'))
    # Phase 2: Pass
    print("Executing Pass...")
    state, player = game.step('pass')
    
    # Check updated yardline
    print(f"New Yardline: {game.yardline}")
    print(f"Type: {type(game.yardline)}")
    
    if isinstance(game.yardline, (int, np.integer)):
        print("[OK] Yardline is integer")
    elif isinstance(game.yardline, float) and game.yardline.is_integer():
        print("[WARN] Yardline is float but discrete value (e.g. 29.0)")
    else:
        print(f"[FAIL] Yardline is float {game.yardline}")
    
    # Check ydstogo
    print(f"YdsToGo: {game.ydstogo}, Type: {type(game.ydstogo)}")

if __name__ == "__main__":
    verify_discrete()
