
import rlcard
import numpy as np

def debug_game_states():
    # Initialize environment with full_game=True (same as training)
    # This disables single_play randomization
    config = {
        'single_play': False, 
        'reward_type': 'epa',
        'seed': 42
    }
    env = rlcard.make('nfl-bucketed', config=config)
    
    print("Simulating 10,000 full drives (full_game=True)...")
    
    very_long_count = 0
    total_steps = 0
    
    # Run loop
    for _ in range(10000):
        state, player_id = env.reset()
        done = False
        while not done:
            # Random action
            action = np.random.choice(list(state['legal_actions'].keys()))
            next_state, next_player_id = env.step(action)
            
            # Check if we hit 1st & Very Long (>15 yds)
            # Down is index 0 in obs, distance is index 1
            # But simpler to look at state dict
            
            down = env.game.down
            ydstogo = env.game.ydstogo
            
            if down == 1 and ydstogo > 15:
                very_long_count += 1
                # print(f"Found 1st & {ydstogo}! Previous Action: {action}")
                
            state = next_state
            player_id = next_player_id
            total_steps += 1
            
            if env.is_over():
                break
                
    print(f"\nSimulation Complete.")
    print(f"Total Steps: {total_steps}")
    print(f"Occurrences of '1st & >15': {very_long_count}")
    print(f"Frequency: {very_long_count/total_steps:.5f}")

if __name__ == "__main__":
    debug_game_states()
