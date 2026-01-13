"""Curriculum training for PPO on various 4th down scenarios."""
import sys
sys.path.insert(0, '.')
import os
import time
import rlcard
import numpy as np
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.games.nfl.game_bucketed import INITIAL_ACTIONS


def train_mixed_scenarios(num_episodes=20000, eval_every=1000, rollout_size=256):
    """Train PPO on randomized mixed 4th down scenarios."""

    print("=" * 60)
    print("4th Down Mixed Scenario Training")
    print("=" * 60)

    # All scenarios to train on (yardline, ydstogo)
    scenarios = [
        # PUNT range (own territory, long distance)
        (25, 10), (20, 10), (30, 10), (15, 10),
        # FG range (opponent territory, any distance)
        (75, 10), (80, 5), (70, 10), (85, 3),
        # GO range (short yardage, anywhere)
        (35, 1), (50, 2), (95, 3), (40, 1),
    ]

    print("Training scenarios:")
    for yardline, ydstogo in scenarios:
        print(f"  4th & {ydstogo} at {yardline}")
    print("=" * 60)

    # Create environment with distribution model
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True,
        'use_distribution_model': True,
    })

    # Create agent with higher entropy for exploration
    agent = PPOAgent(
        state_shape=env.state_shape[0],
        num_actions=7,
        hidden_dims=[128, 128],
        lr=1e-6,
        entropy_coef=0.2,
    )

    # Self-play
    env.set_agents([agent, agent])

    os.makedirs('models/ppo_curriculum', exist_ok=True)

    episode_rewards = []
    step_count = 0
    start_time = time.time()

    for episode in range(1, num_episodes + 1):
        # Randomly pick a scenario
        yardline, ydstogo = scenarios[np.random.randint(len(scenarios))]

        # Reset environment
        state, player_id = env.reset()

        # Set custom 4th down state
        env.game.down = 4
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        state = env.get_state(player_id)

        episode_reward = 0

        while not env.is_over():
            # Get action
            action = agent.step(state)

            # Step environment
            next_state, next_player_id = env.step(action)

            # Handle reward
            if env.is_over():
                payoffs = env.get_payoffs()
                reward = payoffs[player_id]
                agent.feed((state, action, reward, next_state, True))
                episode_reward = payoffs[0]
            else:
                agent.feed((state, action, 0, next_state, False))

            state = next_state
            player_id = next_player_id
            step_count += 1

        episode_rewards.append(episode_reward)

        # Update PPO every rollout_size steps
        if step_count >= rollout_size:
            agent.update()
            step_count = 0

        # Logging
        if episode % eval_every == 0:
            avg_reward = np.mean(episode_rewards[-eval_every:])
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed

            print(f"\n[Episode {episode:,}] Avg EPA: {avg_reward:.3f}, Speed: {eps_per_sec:.1f} eps/sec")
            test_policy(agent, env)

    # Final test
    print("\n" + "=" * 60)
    print("Final Policy Test:")
    test_policy(agent, env)

    # Save model
    agent.save('models/ppo_curriculum/ppo_nfl-bucketed_curriculum.pt')
    print("\nSaved to models/ppo_curriculum/ppo_nfl-bucketed_curriculum.pt")

    return agent


def test_policy(agent, env):
    """Test agent policy on key scenarios."""
    tests = [
        (25, 10, "4th & 10 own 25", "PUNT"),
        (75, 10, "4th & 10 opp 25", "FG"),
        (35, 1, "4th & 1 own 35", "GO"),
        (95, 3, "4th & Goal at 5", "GO"),
    ]

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

        # Consider GO as any formation (not PUNT/FG)
        if expected == "GO":
            is_go = top_name not in ["PUNT", "FG"]
            status = "[OK]" if is_go else "[X]"
        else:
            status = "[OK]" if top_name == expected else "[X]"

        print(f"  {label}: {top_name} ({top_prob:.0f}%) {status} (want {expected})")


if __name__ == '__main__':
    train_mixed_scenarios(num_episodes=5000, eval_every=100, rollout_size=256)
