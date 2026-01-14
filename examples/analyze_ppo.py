"""
Comprehensive PPO Agent Evaluation

Evaluates:
1. 4th Down offensive decisions (PUNT/FG/GO)
2. Defense phase decisions (box count responses)
3. Self-play performance metrics (EPA, win rate)
"""

import argparse
import numpy as np
import torch
import rlcard
from rlcard.agents.ppo_agent import PPOAgent


# Action mappings
OFFENSE_ACTIONS = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY', 'PUNT', 'FG']
DEFENSE_ACTIONS = ['4_box', '5_box', '6_box', '7_box', '8_box']
PLAY_TYPE_ACTIONS = ['Pass', 'Run']


def load_ppo_agent(model_path):
    """Load a trained PPO agent."""
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True, 
        'use_cached_model': True
    })
    
    agent = PPOAgent(
        state_shape=env.state_shape[0], 
        num_actions=7,  # Max actions across phases
        hidden_dims=[128, 128]
    )
    agent.load(model_path)
    return agent, env


def is_go_decision(action_name):
    """Check if the decision is to 'GO' (any formation that's not PUNT/FG)."""
    return action_name not in ['PUNT', 'FG']


# ==================== 4TH DOWN ANALYSIS ====================

FOURTH_DOWN_SCENARIOS = [
    # (yardline, ydstogo, down, label, expected)
    # Clear PUNT situations
    (10, 10, 4, "4th & 10 at own 10", "PUNT"),
    (25, 10, 4, "4th & 10 at own 25", "PUNT"),
    (30, 8, 4, "4th & 8 at own 30", "PUNT"),
    
    # Clear FG situations
    (70, 8, 4, "4th & 8 at opp 30", "FG"),
    (75, 10, 4, "4th & 10 at opp 25", "FG"),
    (85, 8, 4, "4th & 8 at opp 15", "FG"),
    
    # Clear GO situations
    (95, 3, 4, "4th & Goal at 5", "GO"),
    (97, 1, 4, "4th & 1 at opp 3", "GO"),
    (99, 1, 4, "4th & Goal at 1", "GO"),
    
    # Borderline
    (35, 1, 4, "4th & 1 at own 35", "GO"),
    (50, 1, 4, "4th & 1 at midfield", "GO"),
    (60, 1, 4, "4th & 1 at opp 40", "GO"),
]


def analyze_fourth_down(agent, env, verbose=False):
    """Analyze 4th down decisions."""
    print("\n" + "="*70)
    print("PART 1: 4th Down Decision Analysis (Offense Phase 0)")
    print("="*70 + "\n")
    
    correct = 0
    for yardline, ydstogo, down, label, expected in FOURTH_DOWN_SCENARIOS:
        env.reset()
        env.game.down = down
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        
        state = env.get_state(0)
        action, probs = agent.eval_step(state)
        
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        top_action = OFFENSE_ACTIONS[sorted_probs[0][0]]
        
        if expected == "GO":
            is_correct = is_go_decision(top_action)
        else:
            is_correct = (top_action == expected)
        
        if is_correct:
            correct += 1
        
        status = "[OK]" if is_correct else "[X]"
        print(f"{label} (want {expected}):")
        if verbose:
            for act_id, prob in sorted_probs:
                print(f"  {OFFENSE_ACTIONS[act_id]}: {prob*100:.1f}%")
        else:
            for act_id, prob in sorted_probs[:3]:
                print(f"  {OFFENSE_ACTIONS[act_id]}: {prob*100:.1f}%")
        print(f"  Result: {top_action} {status}\n")
    
    total = len(FOURTH_DOWN_SCENARIOS)
    print(f"4th Down Score: {correct}/{total} ({100*correct/total:.1f}%)\n")
    return correct, total


# ==================== DEFENSE ANALYSIS ====================

DEFENSE_SCENARIOS = [
    # (yardline, ydstogo, down, formation, label, expected_tendency)
    # Against I_FORM (run-heavy) - expect more box
    (50, 5, 1, 'I_FORM', "1st & 5 vs I_FORM", "heavy_box"),
    (25, 3, 2, 'I_FORM', "2nd & 3 vs I_FORM", "heavy_box"),
    (75, 1, 3, 'I_FORM', "3rd & 1 vs I_FORM (goal line)", "heavy_box"),
    
    # Against SHOTGUN (pass-heavy) - expect lighter box
    (50, 10, 1, 'SHOTGUN', "1st & 10 vs SHOTGUN", "light_box"),
    (50, 8, 2, 'SHOTGUN', "2nd & 8 vs SHOTGUN", "light_box"),
    (75, 15, 3, 'SHOTGUN', "3rd & 15 vs SHOTGUN", "light_box"),
    
    # Against EMPTY (max pass) - expect lightest box
    (50, 10, 1, 'EMPTY', "1st & 10 vs EMPTY", "light_box"),
    (75, 10, 3, 'EMPTY', "3rd & 10 vs EMPTY", "light_box"),
]


def analyze_defense(agent, env, verbose=False):
    """Analyze defensive decisions based on formation."""
    print("\n" + "="*70)
    print("PART 2: Defense Decision Analysis (Defense Phase 1)")
    print("="*70 + "\n")
    
    formation_to_idx = {'SHOTGUN': 0, 'SINGLEBACK': 1, 'UNDER CENTER': 2, 'I_FORM': 3, 'EMPTY': 4}
    
    results = []
    for yardline, ydstogo, down, formation, label, expected in DEFENSE_SCENARIOS:
        env.reset()
        env.game.down = down
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 1  # Defense phase
        env.game.pending_formation = formation
        
        state = env.get_state(1)  # Player 1 is defense
        action, probs = agent.eval_step(state)
        
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        
        # Calculate average box count
        avg_box = sum(DEFENSE_ACTIONS[act_id].split('_')[0] == str(i+6) and prob * (i+6) 
                      for i, (act_id, prob) in enumerate(probs.items()) 
                      if act_id < len(DEFENSE_ACTIONS))
        
        # Simpler: just get top action
        top_action_idx = sorted_probs[0][0]
        if top_action_idx < len(DEFENSE_ACTIONS):
            top_action = DEFENSE_ACTIONS[top_action_idx]
        else:
            top_action = f"Action_{top_action_idx}"
        
        print(f"{label}:")
        if verbose:
            for act_id, prob in sorted_probs:
                if act_id < len(DEFENSE_ACTIONS):
                    print(f"  {DEFENSE_ACTIONS[act_id]}: {prob*100:.1f}%")
        else:
            for act_id, prob in sorted_probs[:3]:
                if act_id < len(DEFENSE_ACTIONS):
                    print(f"  {DEFENSE_ACTIONS[act_id]}: {prob*100:.1f}%")
        print(f"  Top choice: {top_action}\n")
        
        results.append((formation, top_action, expected))
    
    return results


# ==================== SELF-PLAY METRICS ====================

def run_self_play_evaluation(agent, env, num_games=100):
    """Run self-play games to measure EPA and win rate."""
    print("\n" + "="*70)
    print(f"PART 3: Self-Play Evaluation ({num_games} games)")
    print("="*70 + "\n")
    
    total_rewards = [0, 0]
    wins = [0, 0]
    draws = 0
    total_plays = 0
    
    for game_idx in range(num_games):
        state, player_id = env.reset()
        
        while not env.is_over():
            action, _ = agent.eval_step(state)
            state, player_id = env.step(action)
            total_plays += 1
        
        # Get payoffs
        payoffs = env.get_payoffs()
        total_rewards[0] += payoffs[0]
        total_rewards[1] += payoffs[1]
        
        if payoffs[0] > payoffs[1]:
            wins[0] += 1
        elif payoffs[1] > payoffs[0]:
            wins[1] += 1
        else:
            draws += 1
    
    avg_reward_0 = total_rewards[0] / num_games
    avg_reward_1 = total_rewards[1] / num_games
    avg_plays = total_plays / num_games
    
    print(f"Games played: {num_games}")
    print(f"Average plays per game: {avg_plays:.1f}")
    print(f"Player 0 (Offense) avg reward: {avg_reward_0:.4f}")
    print(f"Player 1 (Defense) avg reward: {avg_reward_1:.4f}")
    print(f"Win rate - P0: {wins[0]/num_games*100:.1f}%, P1: {wins[1]/num_games*100:.1f}%, Draw: {draws/num_games*100:.1f}%")
    print()
    
    # Exploitability proxy: reward difference should be near 0 for balanced play
    reward_diff = abs(avg_reward_0 - avg_reward_1)
    print(f"Reward Imbalance: {reward_diff:.4f} (lower = more balanced)")
    if reward_diff < 0.1:
        print("  -> Good balance between offense and defense\n")
    elif reward_diff < 0.3:
        print("  -> Moderate imbalance\n")
    else:
        print("  -> Significant imbalance (one side dominant)\n")
    
    return {
        'avg_reward_0': avg_reward_0,
        'avg_reward_1': avg_reward_1,
        'win_rate_0': wins[0] / num_games,
        'win_rate_1': wins[1] / num_games,
        'reward_imbalance': reward_diff,
    }


def main():
    parser = argparse.ArgumentParser(description='Comprehensive PPO Agent Evaluation')
    parser.add_argument('model_path', type=str, 
                        help='Path to PPO model')
    parser.add_argument('--verbose', action='store_true',
                        help='Show all action probabilities')
    parser.add_argument('--self-play-games', type=int, default=100,
                        help='Number of self-play games for evaluation')
    parser.add_argument('--skip-self-play', action='store_true',
                        help='Skip self-play evaluation')
    args = parser.parse_args()
    
    print("Loading PPO agent...")
    agent, env = load_ppo_agent(args.model_path)
    
    # Part 1: 4th Down Analysis
    fourth_correct, fourth_total = analyze_fourth_down(agent, env, args.verbose)
    
    # Part 2: Defense Analysis
    defense_results = analyze_defense(agent, env, args.verbose)
    
    # Part 3: Self-Play Metrics
    if not args.skip_self_play:
        metrics = run_self_play_evaluation(agent, env, args.self_play_games)
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"4th Down Accuracy: {fourth_correct}/{fourth_total} ({100*fourth_correct/fourth_total:.1f}%)")
    if not args.skip_self_play:
        print(f"Self-Play Balance: {metrics['reward_imbalance']:.4f} imbalance")
        print(f"Offense Win Rate: {metrics['win_rate_0']*100:.1f}%")


if __name__ == '__main__':
    main()
