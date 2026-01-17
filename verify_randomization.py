from rlcard.games.nfl.game import NFLGame
import numpy as np

print("Verifying 4th Down Randomization...")
# single_play=True triggers the randomization logic
game = NFLGame(single_play=True)
game.start_down = 4

unique_yardlines = set()
unique_ydstogo = set()

for i in range(20):
    game.init_game()
    print(f"Game {i+1}: 4th & {game.ydstogo} at Own {game.yardline}")
    
    # Assertions
    assert 1 <= game.yardline < 100, f"Yardline {game.yardline} out of bounds"
    dist_to_goal = 100 - game.yardline
    max_yds = min(20, dist_to_goal)
    assert 1 <= game.ydstogo <= max_yds + 1, f"YdsToGo {game.ydstogo} invalid for yardline {game.yardline}"
    
    unique_yardlines.add(game.yardline)
    unique_ydstogo.add(game.ydstogo)

print("\nVerification Results:")
print(f"Unique Yardlines observed: {len(unique_yardlines)}")
print(f"Unique YdsToGo observed: {len(unique_ydstogo)}")

if len(unique_yardlines) > 1 and len(unique_ydstogo) > 1:
    print("SUCCESS: Randomization is working.")
else:
    print("FAILURE: Values appear static.")
