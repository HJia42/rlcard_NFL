"""Curriculum training for PPO on various 4th down scenarios."""
import sys
sys.path.insert(0, '.')
import os
import rlcard
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
    
    return results


def main():
    os.makedirs('models/ppo_curriculum', exist_ok=True)
    
    print("=" * 60)
    print("4th Down Curriculum Training")
    print("=" * 60)
    
    # Create environment with distribution model
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True,
        'use_distribution_model': True,
    })
    
    # Create agent
    agent = PPOAgent(
        state_shape=env.state_shape[0],
        num_actions=7,
        hidden_dims=[128, 128],
        lr=3e-4,
        entropy_coef=0.05,
    )
    
    # Curriculum: train on each scenario type sequentially
    scenarios = [
        # Phase 1: Learn PUNT
        [(25, 10), (20, 10), (30, 10)],
        # Phase 2: Learn FG
        [(75, 10), (80, 5), (70, 10)],
        # Phase 3: Learn GO (short yardage)
        [(35, 1), (50, 2), (95, 3)],
    ]
    
    for phase, phase_scenarios in enumerate(scenarios):
        print(f"\n--- Phase {phase + 1} ---")
        for yardline, ydstogo in phase_scenarios:
            print(f"Training on 4th & {ydstogo} at {yardline}...")
            train_scenario(agent, env, yardline, ydstogo, num_episodes=2000)
        
        print("\nTesting policy:")
        test_policy(agent, env)
    
    # Save model
    agent.save('models/ppo_curriculum/ppo_nfl-bucketed_curriculum.pt')
    print("\nSaved to models/ppo_curriculum/ppo_nfl-bucketed_curriculum.pt")


if __name__ == '__main__':
    main()
