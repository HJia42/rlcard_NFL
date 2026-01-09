"""Test NFL game with RLCard agents"""

import sys
sys.path.insert(0, '.')

import rlcard
from rlcard.agents import RandomAgent

# Create environment
env = rlcard.make('nfl', config={'seed': 42})
print(f"NFL Environment created!")
print(f"  Num players: {env.num_players}")
print(f"  Num actions: {env.num_actions}")

# Create random agents
agents = [RandomAgent(num_actions=env.num_actions) for _ in range(env.num_players)]
env.set_agents(agents)

# Run a few games
total_episodes = 10
offense_payoffs = []

for ep in range(total_episodes):
    trajectories, payoffs = env.run()
    offense_payoffs.append(payoffs[0])
    
    if ep < 3:
        print(f"\nEpisode {ep+1}:")
        print(f"  Payoffs: Offense={payoffs[0]:.2f}, Defense={payoffs[1]:.2f}")
        print(f"  Trajectory lengths: {[len(t) for t in trajectories]}")

print(f"\n=== Summary ===")
print(f"Avg Offense Payoff: {sum(offense_payoffs)/len(offense_payoffs):.2f}")
print(f"Min: {min(offense_payoffs):.2f}, Max: {max(offense_payoffs):.2f}")
print("\nNFL game integration successful!")
