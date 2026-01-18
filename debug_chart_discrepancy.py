
import torch
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.utils.agent_loader import load_agent
from rlcard.utils.analysis_utils import get_action_probs, normalize_probs_to_action_names
from rlcard.games.nfl.game import FORMATION_ACTIONS, SPECIAL_TEAMS_ACTIONS

def debug_mismatch():
    model_path = "models/ppo_nfl/ppo_nfl_final.pt"
    game = "nfl"
    
    print(f"Loading agent from {model_path}...")
    agent, env = load_agent('ppo', model_path, game)
    
    # Test Case: 4th & 10 at Own 30
    yardline = 30
    distance = 10
    
    print(f"\nScanning State: 4th & {distance} at Own {yardline}")
    
    # --- method 1: generate_chart logic (replicated) ---
    print("\n--- Method 1: generate_chart.py Logic ---")
    env.reset()
    env.game.down = 4
    env.game.ydstogo = distance
    env.game.yardline = yardline
    env.game.phase = 0
    
    state_chart = env.get_state(0)
    probs_chart = get_action_probs(agent, state_chart)
    
    # Aggregation
    relevant_probs = {}
    all_actions = FORMATION_ACTIONS + SPECIAL_TEAMS_ACTIONS
    for k, v in probs_chart.items():
        if isinstance(k, int):
            if k < len(all_actions):
                relevant_probs[all_actions[k]] = v
        else:
            relevant_probs[k] = v
            
    go_prob = sum(relevant_probs.get(f, 0) for f in FORMATION_ACTIONS)
    punt_prob = relevant_probs.get('PUNT', 0)
    fg_prob = relevant_probs.get('FG', 0)
    
    print(f"Aggregated: GO={go_prob:.4f}, PUNT={punt_prob:.4f}, FG={fg_prob:.4f}")
    if go_prob >= punt_prob and go_prob >= fg_prob:
        print("Chart Decision: GO")
    elif fg_prob >= go_prob and fg_prob >= punt_prob:
        print("Chart Decision: FG")
    else:
        print("Chart Decision: PUNT")

    # --- method 2: analyze_agent logic ---
    print("\n--- Method 2: analyze_agent.py Logic ---")
    # Reset again just to be sure we clear any hidden state
    env.reset()
    env.game.down = 4
    env.game.ydstogo = distance
    env.game.yardline = yardline
    env.game.phase = 0
    
    state_analyze = env.get_state(0)
    probs_analyze = get_action_probs(agent, state_analyze)
    
    # Normalization
    probs_norm = normalize_probs_to_action_names(probs_analyze, FORMATION_ACTIONS + SPECIAL_TEAMS_ACTIONS)
    
    # Sort
    sorted_probs = sorted(probs_norm.items(), key=lambda x: -x[1])
    print("Top actions:")
    for k, v in sorted_probs[:3]:
        print(f"  {k}: {v:.4f}")
        
    top_action = sorted_probs[0][0]
    print(f"Top Action: {top_action}")
    
    if top_action in ['PUNT', 'FG']:
        print(f"Analyze Decision: {top_action}")
    else:
        print("Analyze Decision: GO")

if __name__ == "__main__":
    debug_mismatch()
