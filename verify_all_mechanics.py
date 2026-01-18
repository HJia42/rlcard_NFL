import numpy as np
from rlcard.games.nfl.game import NFLGame
from rlcard.games.nfl.game_iig import NFLGameIIG

def check_integer(val, name, context):
    """Check if value is integer or integer-like float."""
    is_int_type = isinstance(val, (int, np.integer))
    is_int_val = isinstance(val, float) and val.is_integer()
    
    if not (is_int_type or is_int_val):
        print(f"[FAIL] {context}: {name} is {val} (Type: {type(val)})")
        return False
    return True

def test_mechanics(game_cls, name, num_iters=200):
    print(f"\nTesting {name} for {num_iters} iterations...")
    game = game_cls(single_play=True)
    failures = 0
    
    # Test Scrimmage (Pass/Rush)
    for _ in range(num_iters):
        game.init_game()
        
        # Randomize play type
        play_type_idx = np.random.choice([0, 1]) # 0=pass, 1=rush
        
        # Execute play based on game type
        if "IIG" in name:
            # Phase 0: Formation
            game.step(0) # SHOTGUN
            # Phase 1: Play Type (Hidden)
            game.step(play_type_idx)
            # Phase 2: Defense
            game.step(0) # Standard box
        else:
            # Standard: Phase 0 (Formation) -> Phase 1 (Defense) -> Phase 2 (Play Type)
            game.step(0) # SHOTGUN
            game.step(0) # Standard box
            game.step(play_type_idx)
            
        if not check_integer(game.yardline, "Yardline", f"{name} Play"):
            failures += 1
        if not check_integer(game.ydstogo, "YdsToGo", f"{name} Play"):
            failures += 1

    # Test 4th Down Turnover Logic (No Special Teams anymore)
    print(f"Testing 4th Down Turnover logic for {name}...")
    for _ in range(50):
        # Force 4th down state
        game.init_game()
        game.down = 4
        game.ydstogo = 15 # Hard to convert
        
        if "IIG" in name:
            game.step(0)
            game.step(1) # Rush (likely fail)
            game.step(1) # Heavy box
        else:
            game.step(0)
            game.step(1) # Heavy box
            game.step(1) # Rush
            
        yards_gained = game.history[-1]['payoffs'][0] if hasattr(game, 'history') and game.history else 0
        
        # If failure, should be Turnover (is_over=True for single_play, but specifically formatted)
        # In single_play=True, any play is over. 
        # But we want to ensure no crash.
        
    if failures == 0:
        print(f"[PASS] {name}: Logic checks out.")
    else:
        print(f"[FAIL] {name}: {failures} integer violations found.")

if __name__ == "__main__":
    test_mechanics(NFLGame, "NFLGame (Scrimmage Only)")
    test_mechanics(NFLGameIIG, "NFLGameIIG (Scrimmage Only)")
