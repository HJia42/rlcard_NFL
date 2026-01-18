
import os
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.utils.analysis_utils import get_action_probs
from rlcard.games.nfl.game import FORMATION_ACTIONS, SPECIAL_TEAMS_ACTIONS

def load_agent(game_str, model_path, agent_type, device='cpu'):
    # Create valid dummy env for shape
    env = rlcard.make(game_str, config={'single_play': True})
    
    if agent_type == 'deep_cfr':
        # DeepCFRAgent expects model_path to be a directory containing 'model.pt'
        # But we passed the full path. Let's fix it.
        if model_path.endswith('.pt'):
            model_dir = os.path.dirname(model_path)
        else:
            model_dir = model_path
            
        agent = DeepCFRAgent(env, model_path=model_dir, device=device)
        # DeepCFRAgent.load uses torch.load internally. 
        # We might need to monkeypatch or trust its implementation if it doesn't use weights_only=False by default.
        # But looking at source, it just calls torch.load. 
        # If it fails with weights_only, we can't easily fix it without changing agent code.
        # Let's hope torch 2.6 isn't strictly enforcing yet for simple structures, 
        # or that DeepCFR structure is simple enough.
        # UPDATE: We can manually load it here to be safe and set attributes.
        
        path = os.path.join(model_dir, 'model.pt')
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        
        agent.iteration = checkpoint['iteration']
        for i, state_dict in enumerate(checkpoint['advantage_nets']):
            agent.advantage_nets[i].load_state_dict(state_dict)
        agent.policy_net.load_state_dict(checkpoint['policy_net'])
        
        # agent.load() # Skip default load which might fail on weights_only
    elif agent_type == 'nfsp':
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        checkpoint['device'] = device
        agent = NFSPAgent.from_checkpoint(checkpoint)
    elif agent_type == 'ppo':
        # Re-instantiate PPO with same dimensions
        # Assuming [128, 128] hidden dims from training default
        agent = PPOAgent(
            state_shape=env.state_shape[0],
            num_actions=env.num_actions,
            hidden_dims=[128, 128],
            device=device
        )
        agent.load(model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    return agent, env

def get_4th_down_decision(agent, env, yardline, distance, device='cpu'):
    """
    Query agent for 4th Down decision distribution.
    Returns: Dict of {ActionName: Prob}
    """
    # Force state
    env.reset()
    env.game.down = 4
    env.game.ydstogo = distance
    env.game.yardline = yardline
    env.game.phase = 0 # Force offense phase
    
    # Get state for Offense (Player 0)
    state = env.get_state(0)
    
    # Get probabilities
    probs = get_action_probs(agent, state)
    if probs is None:
        return {}
        
    # Filter for Phase 0 actions (Formations + Special Teams)
    # The agent might return probs for ALL actions, but we only care about the initial choice
    # indices 0-6 correspond to FORMATION_ACTIONS + SPECIAL_TEAMS_ACTIONS
    
    relevant_probs = {}
    
    # Convert keys to readable names if they are indices
    all_actions = FORMATION_ACTIONS + SPECIAL_TEAMS_ACTIONS
    
    for k, v in probs.items():
        if isinstance(k, int):
            if k < len(all_actions):
                name = all_actions[k]
                relevant_probs[name] = v
        else:
            # Already string
            relevant_probs[k] = v
            
    # Aggregate "GO" probability vs PUNT vs FG
    # GO = Sum of all Formations
    go_prob = sum(relevant_probs.get(f, 0) for f in FORMATION_ACTIONS)
    punt_prob = relevant_probs.get('PUNT', 0)
    fg_prob = relevant_probs.get('FG', 0)
    
    return {
        'GO': go_prob,
        'PUNT': punt_prob,
        'FG': fg_prob
    }

def generate_heatmap(agent, env, agent_name, output_dir):
    # Grid: Yardline (1-99) x Distance (1-10)
    yardlines = range(1, 100) # 1 to 99
    distances = range(1, 11)  # 1 to 10
    
    # Matrices for each decision
    go_matrix = np.zeros((len(distances), len(yardlines)))
    punt_matrix = np.zeros((len(distances), len(yardlines)))
    fg_matrix = np.zeros((len(distances), len(yardlines)))
    decision_matrix = np.full((len(distances), len(yardlines)), np.nan) # Initialize with NaN
    
    print(f"Sampling {len(yardlines)*len(distances)} states for {agent_name}...")
    
    for i, dist in enumerate(distances):
        for j, yl in enumerate(yardlines):
            # Skip invalid states (e.g. 4th & 10 at 95 yardline -> only 5 yards to goal)
            dist_to_goal = 100 - yl
            if dist > dist_to_goal:
                # Impossible state, leave as NaN
                continue
                
            probs = get_4th_down_decision(agent, env, yl, dist)
            
            go = probs.get('GO', 0)
            punt = probs.get('PUNT', 0)
            fg = probs.get('FG', 0)
            
            go_matrix[i, j] = go
            punt_matrix[i, j] = punt
            fg_matrix[i, j] = fg
            
            # Dominant strategy
            if go >= punt and go >= fg:
                decision_matrix[i, j] = 2 # GO
            elif fg >= go and fg >= punt:
                decision_matrix[i, j] = 1 # FG
            else:
                decision_matrix[i, j] = 0 # PUNT

    # Plotting
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Dominant Decision Map
    plt.figure(figsize=(15, 6))
    # Custom cmap: Blue=Punt, Green=FG, Red=Go
    cmap = sns.color_palette(["#3498db", "#2ecc71", "#e74c3c"]) 
    
    # Mask NaNs (Invalid states)
    mask = np.isnan(decision_matrix)
    
    ax = sns.heatmap(decision_matrix, cmap=cmap, cbar=False, mask=mask,
                     xticklabels=10, yticklabels=distances)
    
    # Invert Y axis so 1 is at top, 10 at bottom? Or standard matrix?
    # Standard matrix: Row 0 is Dist 1. Let's invert y axis to match chart intuition (Short distance at bottom usually?)
    # Actually, standard "NYT 4th Down Bot" charts usually have Distance on Y (1 at top) and Field Pos on X.
    # Let's stick to matrix defaults: Row 0 = Dist 1.
    
    # Fix X axis labels (Yardlines 1, 11, 21...)
    # Current indices are 0..98 corresponding to 1..99
    # xticklabels=10 means every 10th label
    
    plt.title(f"{agent_name}: 4th Down Decision Map (Red=GO, Green=FG, Blue=PUNT)")
    plt.xlabel("Yardline (Own 1 -> Opp 99)")
    plt.ylabel("Yards to Go")
    
    # Custom Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Punt'),
        Patch(facecolor='#2ecc71', label='Field Goal'),
        Patch(facecolor='#e74c3c', label='Go for it')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    
    plt.savefig(os.path.join(output_dir, f"{agent_name}_decision_map.png"))
    plt.close()
    
    print(f"Saved chart to {output_dir}/{agent_name}_decision_map.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True, choices=['deep_cfr', 'nfsp', 'ppo'])
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--game', type=str, default='nfl')
    parser.add_argument('--output', type=str, default='analysis_output')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    
    agent, env = load_agent(args.game, args.model, args.agent, args.device)
    generate_heatmap(agent, env, args.agent, args.output)
