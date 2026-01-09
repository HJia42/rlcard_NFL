"""Train NFSP agent on NFL game"""

import sys
sys.path.insert(0, '.')

import rlcard
from rlcard.agents import NFSPAgent

# Create environment
env = rlcard.make('nfl', config={'seed': 42})
print(f"NFL Environment: {env.num_players} players, {env.num_actions} actions")

# Create NFSP agents for both offense and defense
agents = []
for i in range(env.num_players):
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[i],
        hidden_layers_sizes=[128, 128],
        q_mlp_layers=[128, 128],
        anticipatory_param=0.1,
    )
    agents.append(agent)

env.set_agents(agents)

# Training loop
num_episodes = 1000
eval_every = 100

print(f"\nTraining for {num_episodes} episodes...")

for ep in range(1, num_episodes + 1):
    # Train one episode
    trajectories, payoffs = env.run(is_training=True)
    
    # Feed transitions to agents
    for i in range(env.num_players):
        for ts in zip(trajectories[i][:-1], trajectories[i][1:]):
            agents[i].feed(ts)
    
    # Evaluate
    if ep % eval_every == 0:
        # Run evaluation episodes
        eval_payoffs = []
        for _ in range(50):
            _, p = env.run(is_training=False)
            eval_payoffs.append(p)
        
        avg_offense = sum(p[0] for p in eval_payoffs) / len(eval_payoffs)
        avg_defense = sum(p[1] for p in eval_payoffs) / len(eval_payoffs)
        
        print(f"Episode {ep}: Offense={avg_offense:.2f}, Defense={avg_defense:.2f}")

print("\nTraining complete!")

# Save models
import os
os.makedirs('models', exist_ok=True)
for i, agent in enumerate(agents):
    agent.save(f'models/nfsp_player_{i}.pth')
print("Models saved to models/")
