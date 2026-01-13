"""Test PPO policy consistency across scenarios."""
import sys
sys.path.insert(0, '.')
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.games.nfl.game_bucketed import INITIAL_ACTIONS

env = rlcard.make('nfl-bucketed', config={'single_play': True})

agent = PPOAgent(state_shape=env.state_shape[0], num_actions=7, hidden_dims=[128, 128])
agent.load('models/ppo_nfl/ppo_nfl-bucketed_final.pt')

print('PPO Policy Across Different Scenarios')
print('='*70)

# Test very different states
tests = [
    (25, 10, 1, '1st & 10 own 25'),
    (50, 10, 1, '1st & 10 midfield'),
    (75, 10, 1, '1st & 10 opp 25'),
    (95, 3, 1, '1st & Goal at 5'),
    (25, 10, 4, '4th & 10 own 25'),
    (50, 10, 4, '4th & 10 midfield'),
    (75, 10, 4, '4th & 10 opp 25'),
]

print(f"{'Scenario':<20} | SHOT | SING |  UC  | IFORM| EMPT | PUNT |  FG")
print('-'*70)

for yardline, ydstogo, down, label in tests:
    env.reset()
    env.game.down = down
    env.game.ydstogo = ydstogo
    env.game.yardline = yardline
    env.game.phase = 0
    state = env.get_state(0)
    
    action, probs = agent.eval_step(state)
    p = [probs.get(i, 0)*100 for i in range(7)]
    print(f"{label:<20} | {p[0]:4.1f} | {p[1]:4.1f} | {p[2]:4.1f} | {p[3]:4.1f} | {p[4]:4.1f} | {p[5]:4.1f} | {p[6]:4.1f}")
