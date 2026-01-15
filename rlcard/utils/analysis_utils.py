"""
Unified NFL Agent Analysis Utilities

Shared scenarios, action mappings, and agent-agnostic analysis functions.
Used by analyze_agent.py and compare_agents.py.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


# =============================================================================
# Action Mappings
# =============================================================================

OFFENSE_ACTIONS = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY', 'PUNT', 'FG']
DEFENSE_ACTIONS = ['4_box', '5_box', '6_box', '7_box', '8_box']
PLAY_TYPE_ACTIONS = ['Pass', 'Run']


# =============================================================================
# Test Scenarios
# =============================================================================

# 4th Down scenarios: (yardline, ydstogo, down, label, expected)
FOURTH_DOWN_SCENARIOS = [
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
    
    # Borderline situations
    (35, 1, 4, "4th & 1 at own 35", "GO"),
    (50, 1, 4, "4th & 1 at midfield", "GO"),
    (60, 1, 4, "4th & 1 at opp 40", "GO"),
]

# Defense scenarios: (yardline, ydstogo, down, formation, label, expected_tendency)
DEFENSE_SCENARIOS = [
    # Against I_FORM (run-heavy) - expect heavier box
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

FORMATION_TO_IDX = {'SHOTGUN': 0, 'SINGLEBACK': 1, 'UNDER CENTER': 2, 'I_FORM': 3, 'EMPTY': 4}


# =============================================================================
# Agent-Agnostic Probability Extraction
# =============================================================================

def get_action_probs(agent, state: dict, agent_type: str = None) -> Optional[Dict]:
    """
    Extract action probabilities from any agent type.
    
    Handles differences:
    - PPO: returns {int: float} (action index -> prob)
    - CFR/NFSP/Deep CFR: returns {'action_name': float}
    
    Returns:
        Dict mapping action identifiers to probabilities, or None on error
    """
    try:
        result = agent.eval_step(state)
        
        # Handle different return formats
        if isinstance(result, tuple) and len(result) == 2:
            _, info = result
        else:
            return None
        
        # Extract probabilities
        if isinstance(info, dict):
            if 'probs' in info:
                return info['probs']
            # Some agents return probs directly in info
            return info
        
        return None
    except Exception as e:
        # Print error for debugging state/model mismatches
        import sys
        print(f"Warning: get_action_probs error - {e}", file=sys.stderr)
        return None


def normalize_probs_to_action_names(probs: Dict, action_list: List[str]) -> Dict[str, float]:
    """
    Convert probability dict with int keys to action names.
    
    Args:
        probs: Dict with int or str keys
        action_list: List of action names (e.g., OFFENSE_ACTIONS)
    
    Returns:
        Dict mapping action names to probabilities
    """
    if not probs:
        return {}
    
    # Check if already using string keys
    first_key = next(iter(probs.keys()))
    if isinstance(first_key, str):
        return probs
    
    # Convert int keys to action names
    result = {}
    for idx, prob in probs.items():
        if isinstance(idx, int) and idx < len(action_list):
            result[action_list[idx]] = prob
        else:
            result[f"Action_{idx}"] = prob
    return result


def get_top_action(probs: Dict, action_list: List[str] = None) -> Tuple[str, float]:
    """Get the highest probability action and its probability."""
    if not probs:
        return ("Unknown", 0.0)
    
    top_key = max(probs, key=probs.get)
    top_prob = probs[top_key]
    
    # Convert int key to action name if needed
    if isinstance(top_key, int) and action_list and top_key < len(action_list):
        return (action_list[top_key], top_prob)
    
    return (str(top_key), top_prob)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_fourth_down(agent, env, verbose: bool = False, agent_type: str = None) -> Tuple[int, int]:
    """
    Analyze 4th down decision accuracy.
    
    Args:
        agent: Agent to evaluate
        env: NFL environment
        verbose: Print all action probabilities
        agent_type: Agent type hint for probability extraction
    
    Returns:
        (correct_count, total_count)
    """
    print("\n" + "=" * 70)
    print("4th Down Decision Analysis (Offense Phase 0)")
    print("=" * 70 + "\n")
    
    correct = 0
    for yardline, ydstogo, down, label, expected in FOURTH_DOWN_SCENARIOS:
        env.reset()
        env.game.down = down
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        
        state = env.get_state(0)
        probs = get_action_probs(agent, state, agent_type)
        
        if probs is None:
            print(f"{label}: Could not get probabilities")
            continue
        
        # Normalize to action names
        probs = normalize_probs_to_action_names(probs, OFFENSE_ACTIONS)
        top_action, top_prob = get_top_action(probs, OFFENSE_ACTIONS)
        
        # Check correctness
        if expected == "GO":
            is_correct = top_action not in ['PUNT', 'FG']
        else:
            is_correct = (top_action == expected)
        
        if is_correct:
            correct += 1
        
        status = "[OK]" if is_correct else "[X]"
        print(f"{label} (want {expected}):")
        
        # Sort and display probabilities
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        display_count = len(sorted_probs) if verbose else 3
        for action_name, prob in sorted_probs[:display_count]:
            if prob > 0.01 or verbose:
                print(f"  {action_name}: {prob*100:.1f}%")
        print(f"  Result: {top_action} {status}\n")
    
    total = len(FOURTH_DOWN_SCENARIOS)
    print(f"4th Down Score: {correct}/{total} ({100*correct/total:.1f}%)\n")
    return correct, total


def analyze_defense(agent, env, verbose: bool = False, agent_type: str = None) -> List[Tuple]:
    """
    Analyze defensive box count decisions based on offensive formation.
    
    Args:
        agent: Agent to evaluate
        env: NFL environment
        verbose: Print all action probabilities
        agent_type: Agent type hint
    
    Returns:
        List of (formation, top_action, expected) tuples
    """
    print("\n" + "=" * 70)
    print("Defense Decision Analysis (Defense Phase 1)")
    print("=" * 70 + "\n")
    
    results = []
    for yardline, ydstogo, down, formation, label, expected in DEFENSE_SCENARIOS:
        env.reset()
        env.game.down = down
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 1
        env.game.pending_formation = formation
        
        state = env.get_state(1)  # Player 1 is defense
        probs = get_action_probs(agent, state, agent_type)
        
        if probs is None:
            print(f"{label}: Could not get probabilities")
            continue
        
        probs = normalize_probs_to_action_names(probs, DEFENSE_ACTIONS)
        top_action, top_prob = get_top_action(probs, DEFENSE_ACTIONS)
        
        print(f"{label}:")
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        display_count = len(sorted_probs) if verbose else 3
        for action_name, prob in sorted_probs[:display_count]:
            if prob > 0.01 or verbose:
                print(f"  {action_name}: {prob*100:.1f}%")
        print(f"  Top choice: {top_action}\n")
        
        results.append((formation, top_action, expected))
    
    return results


def run_self_play_evaluation(agent, env, num_games: int = 100) -> Dict[str, float]:
    """
    Run self-play games and return performance metrics.
    
    Args:
        agent: Agent to evaluate
        env: NFL environment
        num_games: Number of games to run
    
    Returns:
        Dict with avg_reward_0, avg_reward_1, win_rate_0, win_rate_1, reward_imbalance
    """
    print("\n" + "=" * 70)
    print(f"Self-Play Evaluation ({num_games} games)")
    print("=" * 70 + "\n")
    
    env.set_agents([agent, agent])
    
    total_rewards = [0, 0]
    wins = [0, 0]
    draws = 0
    total_plays = 0
    
    for _ in range(num_games):
        state, player_id = env.reset()
        
        while not env.is_over():
            action, _ = agent.eval_step(state)
            state, player_id = env.step(action)
            total_plays += 1
        
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
    reward_diff = abs(avg_reward_0 - avg_reward_1)
    
    print(f"Games played: {num_games}")
    print(f"Average plays per game: {avg_plays:.1f}")
    print(f"Player 0 (Offense) avg EPA: {avg_reward_0:.4f}")
    print(f"Player 1 (Defense) avg EPA: {avg_reward_1:.4f}")
    print(f"Win rate - P0: {wins[0]/num_games*100:.1f}%, P1: {wins[1]/num_games*100:.1f}%, Draw: {draws/num_games*100:.1f}%")
    print(f"\nReward Imbalance: {reward_diff:.4f} (lower = more balanced)")
    
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
