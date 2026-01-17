import torch
import sys
import os
sys.path.insert(0, '.')
from rlcard.agents import NFSPAgent
from rlcard.games.nfl.game import NFLGame
import rlcard

# Load model
model_path = 'models/test_nfsp_analysis/nfsp_nfl_p0_final.pt'
print(f"Loading {model_path}...")
checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
agent = NFSPAgent.from_checkpoint(checkpoint)

# Init info
env = rlcard.make('nfl')
state, _ = env.reset()

print("State keys:", state.keys())
print("raw_legal_actions:", state.get('raw_legal_actions'))
print("legal_actions:", state.get('legal_actions'))

# Get action
print("\nQuerying agent...")
action, info = agent.eval_step(state)

print(f"Action: {action}")
if 'probs' in info:
    print(f"Raw Probs: {info['probs']}")
