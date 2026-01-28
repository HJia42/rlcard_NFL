
import os
import argparse
import numpy as np
import pandas as pd
import torch
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.games.nfl.game_bucketed import NFLGameBucketed, FORMATION_ACTIONS

# Mock buckets for reconstruction
DISTANCE_MAP = {
    0: 2,   # Short (1-3) -> 2
    1: 5,   # Medium (4-7) -> 5
    2: 10,  # Long (8-15) -> 10
    3: 20   # Very Long (16+) -> 20
}

FIELD_MAP = {i: i*5 + 3 for i in range(20)} # Midpoints (3, 8, 13...)

def load_agent(env_name, agent_type, device='cpu'):
    """Load trained agent (reused logic)."""
    env = rlcard.make(env_name, config={'single_play': True})
    
    suffix = "std" if "bucketed" == env_name or "nfl-bucketed" == env_name else "iig"
    if "iig" in env_name: suffix = "iig"
    
    dir_name = f"nfl_score_full_{agent_type}_{suffix}"
    base_dir = os.path.join("experiments", dir_name)
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    if not os.path.exists(checkpoint_dir) and not os.path.exists(base_dir):
        print(f"Skipping {agent_type} (not found at {base_dir})")
        return None

    # Fix generic state shape issue
    # Agent expects flat state shape usually, or handles it internally.
    # PPOAgent init handles list/tuple.
    # But check state_shape structure
    if isinstance(env.state_shape, list) and isinstance(env.state_shape[0], list):
        agent_state_shape = env.state_shape[0]
    else:
        agent_state_shape = env.state_shape

    try:
        if agent_type == 'ppo':
            agent = PPOAgent(agent_state_shape, env.num_actions, hidden_dims=[128, 128], device=device)
            path = os.path.join(checkpoint_dir, 'agent_0.pt')
            if os.path.exists(path): agent.load(path)
            
        elif agent_type == 'nfsp':
            agent = NFSPAgent(env.num_actions, agent_state_shape, hidden_layers_sizes=[128, 128], q_mlp_layers=[128, 128], device=device)
            path = os.path.join(checkpoint_dir, 'agent_0.pt')
            if os.path.exists(path): agent.load(path)
            
        elif agent_type == 'deep_cfr':
            agent = DeepCFRAgent(env, device=device, hidden_layers=[128, 128])
            path = os.path.join(base_dir, 'model.pt') # Try base first
            if not os.path.exists(path): path = os.path.join(checkpoint_dir, 'model.pt')
            if os.path.exists(path):
                agent.model_path = os.path.dirname(path)
                agent.load()
        return agent
    except Exception as e:
        print(f"Error loading {agent_type}: {e}")
        return None

def force_state(env, down, dist_bucket, field_bucket, phase, pending_formation=None):
    """Force environment into specific bucket state."""
    game = env.game
    game.down = down
    game.ydstogo = DISTANCE_MAP[dist_bucket]
    game.yardline = FIELD_MAP[field_bucket]
    game.phase = phase
    game.pending_formation = pending_formation
    game.pending_defense_action = None # Reset defense
    
    # Recalculate legal actions 
    # (Private method access usually needed, but for bucketed env actions are static per phase usually)
    # Actually get_state calls logic that might depend on this?
    # In nfl game, legal actions depend on phase.
    
    return env.get_state(0) # Player 0 is Offense

def analyze_agent(env_name, agent_type, device):
    agent = load_agent(env_name, agent_type, device)
    if not agent: return []
    
    if hasattr(agent, 'set_mode'): agent.set_mode(rlcard.agents.Agent.EVAL)
    
    env = rlcard.make(env_name, config={'single_play': True})
    results = []
    
    print(f"Analyzing {agent_type} in {env_name}...")
    
    # Iterate all buckets
    # Down: 1-4
    # Dist: 0-3
    # Field: 0-19
    for down in range(1, 5):
        for dist in range(4):
            for field in range(20):
                
                # --- Phase 0: Formation Selection ---
                state_p0 = force_state(env, down, dist, field, phase=0)
                _, info = agent.eval_step(state_p0)
                form_probs = info.get('probs', {})
                
                # Normalize just in case 
                # (actions like 'punt', 'field_goal' might be available on 4th down)
                # We want to identify top formation and weighted pass prob
                
                total_pass_prob = 0.0
                total_run_prob = 0.0
                weighted_punt = 0.0
                weighted_fg = 0.0
                
                # For each formation, check Plan (Pass/Run) probability
                for form_action, f_prob in form_probs.items():
                    if f_prob < 0.001: continue 
                    
                    if form_action in ['punt', 'field_goal']:
                        if form_action == 'punt': weighted_punt += f_prob
                        if form_action == 'field_goal': weighted_fg += f_prob
                        continue
                        
                    # If action is a formation, simulate next step
                    if form_action in FORMATION_ACTIONS:
                        # Determine Play Phase Index
                        # Standard: Phase 2. IIG: Phase 1.
                        play_phase = 1 if 'iig' in env_name else 2
                        
                        state_play = force_state(env, down, dist, field, phase=play_phase, pending_formation=form_action)
                        _, info_play = agent.eval_step(state_play)
                        play_probs = info_play.get('probs', {})
                        
                        # Accumulate
                        p_pass = play_probs.get('pass', 0.0)
                        p_run = play_probs.get('rush', 0.0)
                        
                        total_pass_prob += f_prob * p_pass
                        total_run_prob += f_prob * p_run

                # Get Top Formation
                if form_probs:
                    top_form = max(form_probs, key=form_probs.get)
                    top_form_prob = form_probs[top_form]
                else:
                    top_form = 'None'
                    top_form_prob = 0.0
                
                # Filter Impossible States (No Penalties = No 1st & >10)
                # Dist 2 = Long (8-15), Dist 3 = Very Long (16+)
                if down == 1 and dist >= 2:
                    continue

                results.append({
                    'Env': env_name,
                    'Agent': agent_type,
                    'Down': down,
                    'Dist': ['Short', 'Med', 'Long', 'V.Long'][dist],
                    'Field': f"Own {FIELD_MAP[field]}" if field < 10 else f"Opp {100-FIELD_MAP[field]}",
                    'Top_Choice': top_form,
                    'Top_Prob': round(top_form_prob, 3),
                    'Pass_Prob': round(total_pass_prob, 3),
                    'Run_Prob': round(total_run_prob, 3),
                    'Punt_Prob': round(weighted_punt, 3),
                    'FG_Prob': round(weighted_fg, 3)
                })
                
    return results

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    all_results = []
    
    envs = ['nfl-bucketed', 'nfl-iig-bucketed']
    agents = ['ppo', 'nfsp', 'deep_cfr']
    
    for env in envs:
        for agent in agents:
            res = analyze_agent(env, agent, device)
            all_results.extend(res)
            
    df = pd.DataFrame(all_results)
    
    # Save Full CSV
    csv_path = 'agent_policy_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nFull analysis saved to {csv_path}")
    
    # Print Summary (Aggregated by Down)
    print("\n=== Aggregated Strategy by Agent & Down ===")
    if not df.empty:
        summary = df.groupby(['Env', 'Agent', 'Down'])[['Pass_Prob', 'Run_Prob', 'Punt_Prob', 'FG_Prob']].mean()
        print(summary)
    else:
        print("No results found.")

if __name__ == '__main__':
    main()
