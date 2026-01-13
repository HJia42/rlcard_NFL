"""Curriculum training for PPO on various 4th down scenarios."""
import sys
sys.path.insert(0, '.')
import os
import time
import rlcard
import numpy as np
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.games.nfl.game_bucketed import INITIAL_ACTIONS


def train_scenario(agent, env, yardline, ydstogo, num_episodes=2500):
    """Train on a specific 4th down scenario."""
    
    for episode in range(num_episodes):
        # Reset and set custom state
        env.reset()
        env.game.down = 4
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        
        # Run episode with self-play
        state = env.get_state(0)
        
        while not env.is_over():
            player_id = env.get_player_id()
            action = agent.step(state) if player_id == 0 else agent.eval_step(state)[0]
            next_state, next_player = env.step(action)
            
            if not env.is_over():
                state = env.get_state(next_player)
        
        # Feed reward
        reward = env.get_payoffs()[0]
        agent.feed_reward(reward)
    
    return agent


def test_policy(agent, env):
    """Test agent policy on key scenarios."""
    tests = [
        (25, 10, "4th & 10 own 25", "PUNT"),
        (75, 10, "4th & 10 opp 25", "FG"),
        (35, 1, "4th & 1 own 35", "GO"),
        (95, 3, "4th & Goal at 5", "GO"),
    ]
    
    results = []
    for yardline, ydstogo, label, expected in tests:
        env.reset()
        env.game.down = 4
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        state = env.get_state(0)

        action, probs = agent.eval_step(state)

        top_action = max(probs, key=probs.get)
        top_name = INITIAL_ACTIONS[top_action]
        top_prob = probs[top_action] * 100
        
        status = "✓" if top_name == expected else "✗"
        results.append((label, top_name, top_prob, expected, status))
        print(f"  {label}: {top_name} ({top_prob:.0f}%) {status} (want {expected})")


if __name__ == '__main__':
    train_mixed_scenarios(num_episodes=5000, eval_every=100, rollout_size=256)
