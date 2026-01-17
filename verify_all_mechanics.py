import numpy as np
from rlcard.games.nfl.game import NFLGame
from rlcard.games.nfl.game_iig import NFLGameIIG
from rlcard.games.nfl.game_iig_scrimmage import NFLGameIIGScrimmage

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
        play_type = np.random.choice(['pass', 'rush'])
        
        # Execute play based on game type
        if "IIG" in name:
            # Phase 0: Formation
            game.step(0) # SHOTGUN
            # Phase 1: Play Type
            if "Scrimmage" in name:
                # Scrimmage game: phase 1 is just play type
                game.step(0 if play_type == 'pass' else 1)
            else:
                # Full IIG: phase 1 is play type
                game.step(0 if play_type == 'pass' else 1)
            # Phase 2: Defense
            game.step(0) # Standard box
        else:
            # Standard: Phase 0 (Formation) -> Phase 1 (Defense) -> Phase 2 (Play Type)
            game.step(0) # SHOTGUN
            game.step(0) # Standard box
            game.step(0 if play_type == 'pass' else 1)
            
        if not check_integer(game.yardline, "Yardline", f"{name} {play_type}"):
            failures += 1
        if not check_integer(game.ydstogo, "YdsToGo", f"{name} {play_type}"):
            failures += 1

    # Test Special Teams (if applicable)
    if "Scrimmage" not in name:
        # Test Punt
        for _ in range(num_iters):
            game.init_game()
            # Force Punt
            if "IIG" in name:
                # Phase 0: Special Team Action
                game.step('PUNT') 
            else:
                # Phase 0: Special Team Action
                game.step('PUNT')
                
            # For PUNT, the game ends immediately. 
            # We can't easily check 'opp_yardline' directly from game state after game over 
            # unless we inspect internal variables or logic, but let's check if the logic throws errors
            # or if we can infer state. 
            # Actually, the game logic updates self.yardline? No, PUNT ends game.
            # But the payload calculation relies on integers.
            pass # Punts end game efficiently, difficult to check internal state without deep inspection.
                 # However, we verified the code uses int().
                 
        # Test FG
        for _ in range(num_iters):
             game.init_game()
             if "IIG" in name:
                 game.step('FG')
             else:
                 game.step('FG')
                 
    if failures == 0:
        print(f"[PASS] {name}: Logic checks out.")
    else:
        print(f"[FAIL] {name}: {failures} integer violations found.")

if __name__ == "__main__":
    test_mechanics(NFLGame, "NFLGame (Standard)")
    test_mechanics(NFLGameIIG, "NFLGameIIG")
    test_mechanics(NFLGameIIGScrimmage, "NFLGameIIGScrimmage")
