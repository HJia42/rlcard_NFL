import numpy as np
import rlcard
from rlcard.games.nfl.game_iig import NFLGameIIG

def verify_discrete_iig():
    print("Initializing NFLGameIIG...")
    game = NFLGameIIG(single_play=False)
    game.init_game()
    game.yardline = 25
    game.ydstogo = 10
    
    print(f"Start: Yardline={game.yardline}, Type={type(game.yardline)}")
    
    # Simulate a fake IIG play
    # Phase 0: Formation
    print("Phase 0: Offense chooses SHOTGUN")
    game.step('SHOTGUN')
    
    # Phase 1: Play Type (Hidden/Committed)
    print("Phase 1: Offense commits to PASS")
    game.step('pass')
    
    # Phase 2: Defense
    print("Phase 2: Defense chooses Box Count 6")
    # Find index for (6, 'Standard')
    from rlcard.games.nfl.game import DEFENSE_ACTIONS
    action_idx = DEFENSE_ACTIONS.index((6, 'Standard'))
    # This triggers outcome resolution
    state, player = game.step(action_idx)
    
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
    verify_discrete_iig()
