"""Compare specialist vs generalist PPO agents for 4th down decisions."""
import sys
sys.path.insert(0, '.')
import os
import time
import rlcard
import numpy as np
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.games.nfl.game_bucketed import INITIAL_ACTIONS


# Test scenarios - each specialist trains on ONE of these
TEST_SCENARIOS = [
    (25, 10, "4th & 10 own 25", "PUNT"),
    (75, 10, "4th & 10 opp 25", "FG"),
    (35, 1, "4th & 1 own 35", "GO"),
    (95, 3, "4th & Goal at 5", "GO"),
]


def train_agent(env, scenarios, num_episodes=5000, rollout_size=256, name="agent"):
    """Train PPO agent on given scenarios."""
    
    agent = PPOAgent(
        state_shape=env.state_shape[0],
        num_actions=7,
        hidden_dims=[128, 128],
        lr=1e-3,
        entropy_coef=0.05,
    )
    
    env.set_agents([agent, agent])
    step_count = 0
    
    for episode in range(1, num_episodes + 1):
        # Pick scenario
        yardline, ydstogo = scenarios[np.random.randint(len(scenarios))]
        
        # Reset and set state
        state, player_id = env.reset()
        env.game.down = 4
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        state = env.get_state(player_id)
        
        while not env.is_over():
            action = agent.step(state)
            next_state, next_player_id = env.step(action)
            
            if env.is_over():
                payoffs = env.get_payoffs()
                agent.feed((state, action, payoffs[player_id], next_state, True))
            else:
                agent.feed((state, action, 0, next_state, False))
            
            state = next_state
            player_id = next_player_id
            step_count += 1
        
        if step_count >= rollout_size:
            agent.update()
            step_count = 0
        
        if episode % 1000 == 0:
            print(f"  {name}: Episode {episode:,}")
    
    return agent


def test_agent(agent, env, scenarios):
    """Test agent on all scenarios, return dict of results."""
    results = {}
    
    for yardline, ydstogo, label, expected in scenarios:
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
        
        # Check if correct
        if expected == "GO":
            is_correct = top_name not in ["PUNT", "FG"]
        else:
            is_correct = top_name == expected
        
        results[label] = {
            'action': top_name,
            'prob': top_prob,
            'expected': expected,
            'correct': is_correct
        }
    
    return results


def main():
    print("=" * 70)
    print("SPECIALIST vs GENERALIST PPO COMPARISON")
    print("=" * 70)
    
    # Create environment
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True,
        'use_distribution_model': True,
    })
    
    # All training scenarios for generalist
    all_scenarios = [
        (25, 10), (20, 10), (30, 10), (15, 10),  # PUNT
        (75, 10), (80, 5), (70, 10), (85, 3),    # FG
        (35, 1), (50, 2), (95, 3), (40, 1),      # GO
    ]
    
    specialist_episodes = 3000
    generalist_episodes = 15000
    
    # Train 4 specialists
    print("\n--- Training 4 Specialist Agents ---")
    specialists = {}
    for yardline, ydstogo, label, expected in TEST_SCENARIOS:
        print(f"\nTraining specialist for: {label}")
        agent = train_agent(
            env, 
            [(yardline, ydstogo)],  # Only one scenario
            num_episodes=specialist_episodes,
            name=label
        )
        specialists[label] = agent
    
    # Train generalist
    print("\n--- Training Generalist Agent ---")
    generalist = train_agent(
        env,
        all_scenarios,
        num_episodes=generalist_episodes,
        name="Generalist"
    )
    
    # Test all agents
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Header
    print(f"\n{'Scenario':<22} | {'Specialist':<20} | {'Generalist':<20}")
    print("-" * 70)
    
    # Test each scenario
    for yardline, ydstogo, label, expected in TEST_SCENARIOS:
        # Specialist (use the one trained on this scenario)
        spec_results = test_agent(specialists[label], env, [(yardline, ydstogo, label, expected)])
        spec_r = spec_results[label]
        spec_str = f"{spec_r['action']} ({spec_r['prob']:.0f}%)"
        spec_mark = "[OK]" if spec_r['correct'] else "[X]"
        
        # Generalist
        gen_results = test_agent(generalist, env, [(yardline, ydstogo, label, expected)])
        gen_r = gen_results[label]
        gen_str = f"{gen_r['action']} ({gen_r['prob']:.0f}%)"
        gen_mark = "[OK]" if gen_r['correct'] else "[X]"
        
        print(f"{label:<22} | {spec_str:<15} {spec_mark:<4} | {gen_str:<15} {gen_mark:<4}")
    
    # Summary
    print("\n" + "-" * 70)
    spec_correct = sum(1 for label in specialists 
                       for s in [test_agent(specialists[label], env, 
                                           [(y, d, l, e) for y, d, l, e in TEST_SCENARIOS if l == label])]
                       if list(s.values())[0]['correct'])
    
    gen_results = test_agent(generalist, env, TEST_SCENARIOS)
    gen_correct = sum(1 for r in gen_results.values() if r['correct'])
    
    print(f"\nSpecialists: {spec_correct}/4 correct (each on its own scenario)")
    print(f"Generalist:  {gen_correct}/4 correct (on all scenarios)")
    
    # Save generalist
    os.makedirs('models/ppo_curriculum', exist_ok=True)
    generalist.save('models/ppo_curriculum/ppo_generalist.pt')
    print("\nSaved generalist to models/ppo_curriculum/ppo_generalist.pt")


if __name__ == '__main__':
    main()
