"""Train NFSP agents on NFL game

Usage:
    python examples/nfl_nfsp_train.py
"""

import sys
sys.path.insert(0, '.')

import os
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.utils import reorganize

# Create environment
env = rlcard.make('nfl', config={'seed': 42})
eval_env = rlcard.make('nfl', config={'seed': 43})

print(f"NFL Environment: {env.num_players} players, {env.num_actions} actions")

# Create NFSP agents for both offense and defense
agents = []
for i in range(env.num_players):
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=[11],  # Fixed state size (padded)
        hidden_layers_sizes=[128, 128],
        q_mlp_layers=[128, 128],
        anticipatory_param=0.1,
    )
    agents.append(agent)

env.set_agents(agents)
eval_env.set_agents(agents)

# Training parameters
num_episodes = 10000
eval_every = 500
save_every = 2000

print(f"\nTraining NFSP for {num_episodes} episodes...")

for ep in range(1, num_episodes + 1):
    # Train one episode
    trajectories, payoffs = env.run(is_training=True)
    
    # Reorganize trajectories into proper format
    trajectories = reorganize(trajectories, payoffs)
    
    # Feed transitions to agents
    for i in range(env.num_players):
        for transition in trajectories[i]:
            agents[i].feed(transition)
    
    # Evaluate periodically
    if ep % eval_every == 0:
        eval_payoffs = []
        for _ in range(50):
            _, p = eval_env.run(is_training=False)
            eval_payoffs.append(p)
        
        avg_offense = sum(p[0] for p in eval_payoffs) / len(eval_payoffs)
        print(f"Episode {ep}: Avg Offense EPA = {avg_offense:.2f}")
    
    # Save periodically
    if ep % save_every == 0:
        os.makedirs('models/nfl', exist_ok=True)
        for i, agent in enumerate(agents):
            agent.save_checkpoint('models/nfl', filename=f'nfsp_player_{i}_{ep}.pt')
        print(f"Saved models at episode {ep}")

print("\nTraining complete!")

# Final save
os.makedirs('models/nfl', exist_ok=True)
for i, agent in enumerate(agents):
    agent.save_checkpoint('models/nfl', filename=f'nfsp_player_{i}_final.pt')
print("Final models saved to models/nfl/")
