"""
Analyze PPO Agent - 4th Down Decision Analysis

Tests the PPO agent on various 4th down scenarios to evaluate decision quality.
"""

import argparse
import torch
import rlcard
from rlcard.agents.ppo_agent import PPOAgent


ACTIONS = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY', 'PUNT', 'FG']

# Test scenarios: (yardline, ydstogo, down, label, expected_action)
# yardline: 1-50 = own territory, 51-99 = opponent territory
SCENARIOS = [
    # Clear PUNT situations (deep in own territory)
    (10, 10, 4, "4th & 10 at own 10", "PUNT"),
    (20, 15, 4, "4th & 15 at own 20", "PUNT"),
    (25, 10, 4, "4th & 10 at own 25", "PUNT"),
    (30, 8, 4, "4th & 8 at own 30", "PUNT"),
    
    # Clear FG situations (in FG range)
    (65, 10, 4, "4th & 10 at opp 35 (FG range)", "FG"),
    (70, 8, 4, "4th & 8 at opp 30", "FG"),
    (75, 10, 4, "4th & 10 at opp 25", "FG"),
    (80, 5, 4, "4th & 5 at opp 20", "FG"),
    
    # Clear GO situations (short yardage or goal line)
    (95, 1, 4, "4th & 1 at opp 5 (goal line)", "GO"),
    (95, 3, 4, "4th & Goal at 5", "GO"),
    (97, 1, 4, "4th & 1 at opp 3", "GO"),
    (99, 1, 4, "4th & Goal at 1", "GO"),
    
    # Borderline decisions (interesting to analyze)
    (35, 1, 4, "4th & 1 at own 35", "GO"),  # Short yardage but deep
    (40, 2, 4, "4th & 2 at own 40", "GO"),  # Short yardage at midfield
    (50, 1, 4, "4th & 1 at midfield", "GO"),  # Midfield short yardage
    (55, 3, 4, "4th & 3 at opp 45", "PUNT"),  # Too far for FG, medium distance
    (60, 1, 4, "4th & 1 at opp 40", "GO"),  # Short yardage in opponent territory
    (85, 8, 4, "4th & 8 at opp 15", "FG"),  # Long for TD, easy FG
    
    # Non-4th down (should always GO)
    (25, 10, 1, "1st & 10 at own 25", "GO"),
    (50, 5, 2, "2nd & 5 at midfield", "GO"),
    (75, 3, 3, "3rd & 3 at opp 25", "GO"),
]


def load_ppo_agent(model_path):
    """Load a trained PPO agent."""
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True, 
        'use_cached_model': True
    })
    
    agent = PPOAgent(
        state_shape=env.state_shape[0], 
        num_actions=7, 
        hidden_dims=[128, 128]
    )
    agent.load(model_path)
    return agent, env


def evaluate_scenario(agent, env, yardline, ydstogo, down):
    """Evaluate agent's decision for a given scenario."""
    env.reset()
    env.game.down = down
    env.game.ydstogo = ydstogo
    env.game.yardline = yardline
    env.game.phase = 0
    
    state = env.get_state(0)
    action, probs = agent.eval_step(state)
    
    return probs


def is_go_decision(top_action):
    """Check if the decision is to 'GO' (any formation that's not PUNT/FG)."""
    return top_action not in ['PUNT', 'FG']


def main():
    parser = argparse.ArgumentParser(description='Analyze PPO Agent on 4th Down Scenarios')
    parser.add_argument('model_path', type=str, 
                        help='Path to PPO model (e.g., models_cloud/ppo_nfl-bucketed_final.pt)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all action probabilities')
    args = parser.parse_args()
    
    print("Loading PPO agent...")
    agent, env = load_ppo_agent(args.model_path)
    
    print("\n" + "="*70)
    print("PPO Agent - 4th Down Decision Analysis")
    print("="*70 + "\n")
    
    correct = 0
    total = 0
    
    for yardline, ydstogo, down, label, expected in SCENARIOS:
        probs = evaluate_scenario(agent, env, yardline, ydstogo, down)
        
        # Sort by probability
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        top_action = ACTIONS[sorted_probs[0][0]]
        top_prob = sorted_probs[0][1]
        
        # Determine if correct
        if expected == "GO":
            is_correct = is_go_decision(top_action)
        else:
            is_correct = (top_action == expected)
        
        if is_correct:
            correct += 1
        total += 1
        
        # Print result
        status = "[OK]" if is_correct else "[X]"
        print(f"{label} (want {expected}):")
        
        if args.verbose:
            for act_id, prob in sorted_probs:
                marker = "<--" if ACTIONS[act_id] == expected or (expected == "GO" and is_go_decision(ACTIONS[act_id])) else ""
                print(f"  {ACTIONS[act_id]}: {prob*100:.1f}% {marker}")
        else:
            # Show top 3
            for act_id, prob in sorted_probs[:3]:
                print(f"  {ACTIONS[act_id]}: {prob*100:.1f}%")
        
        print(f"  Result: {top_action} {status}\n")
    
    print("="*70)
    print(f"Final Score: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("="*70)


if __name__ == '__main__':
    main()
