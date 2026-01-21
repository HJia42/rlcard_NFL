
import os
import sys
import argparse
import numpy as np
import torch
import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import tournament

def load_agent(env_name, agent_type, device='cpu'):
    """Load a trained agent from the default experiment path."""
    env = rlcard.make(env_name, config={'single_play': True})
    
    # Construct default path based on training script convention
    # experiments/{env}_{agent}/checkpoints/
    base_dir = f'experiments/{env_name}_{agent_type}'
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    
    if not os.path.exists(checkpoint_dir):
        print(f"  [Warn] No checkpoint dir found at {checkpoint_dir}")
        return None

    # Sanitize State Shape to avoid "144 dimension" bug
    if isinstance(env.state_shape, list) and isinstance(env.state_shape[0], list):
        agent_state_shape = env.state_shape[0]
    else:
        agent_state_shape = env.state_shape

    try:
        if agent_type == 'ppo':
            agent = PPOAgent(
                state_shape=agent_state_shape,
                num_actions=env.num_actions,
                hidden_dims=[128, 128],
                device=device
            )
            # PPO typically saves agent_0.pt (offense) and agent_1.pt (defense)
            # We want the offense agent for offense tests, defense for defense tests.
            # But the object handles both usually? No, RLCard PPO is single-agent usually or self-play.
            # In our training, we have a list of agents [PPO, PPO].
            # Let's load Player 0 (Offense) by default.
            path = os.path.join(checkpoint_dir, 'agent_0.pt')
            agent.load(path) # Fix: Pass path string not loaded valid dict
            
        elif agent_type == 'nfsp':
            # Initialize agent before loading
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=agent_state_shape,
                hidden_layers_sizes=[128, 128],
                q_mlp_layers=[128, 128],
                device=device
            )
            # NFSP load expects the directory path
            path = os.path.join(checkpoint_dir, 'agent_0.pt')
            try:
                agent.load(path)
            except Exception as e:
                print(f"  [Error] NFSP load failed: {e}")
                return None
            
        elif agent_type == 'deep_cfr':
            agent = DeepCFRAgent(
                env,
                device=device
            )
            path = os.path.join(checkpoint_dir, 'model.pt')
            if not os.path.exists(path):
                # Fallback to root experiment dir (where 'Save Final' puts it)
                path = os.path.join(base_dir, 'model.pt')
            
            if os.path.exists(path):
                # Correct loading for DeepCFR:
                # 1. Provide the directory containing 'model.pt'
                agent.model_path = os.path.dirname(path)
                # 2. Call load() with NO arguments
                if not agent.load():
                     print(f"  [Error] DeepCFR internal load failed for {path}")
                     return None
            else:
                print(f"  [Debug] DeepCFR model not found at:\n    1. {os.path.join(checkpoint_dir, 'model.pt')}\n    2. {path}")
                return None
            
        else:
            return None
            
        return agent
        
    except Exception as e:
        print(f"  [Error] Failed to load {agent_type} from {checkpoint_dir}: {e}")
        return None

def extract_payoff(payoffs, player_id):
    """
    Robustly extract payoff for a specific player.
    Handles standard [p0, p1] and nested [array([p0, p1]), 0.0] cases.
    """
    # Check if first element is an array/list that looks like it holds all payoffs
    p0_val = payoffs[0]
    
    # Case: IIG-like structure where payoffs[0] is array([p0_avg, p1_avg])
    if isinstance(p0_val, (list, np.ndarray, torch.Tensor)):
        if len(p0_val) >= 2:
            val = float(p0_val[player_id])
            return val
            
    # Standard Case: payoffs[i] is the scalar for player i
    return float(payoffs[player_id])

def evaluate_margin(agent, env_name, num_games):
    """
    Calculate EPA Margin against Standard Random Baseline.
    Returns: (Offense Margin, Defense Margin)
    """
    env = rlcard.make(env_name, config={'single_play': True})
    
    # 1. Agent vs Random (Agent is Offense/Player 0)
    # -----------------------------------------------
    random_agent = RandomAgent(num_actions=env.num_actions)
    env.set_agents([agent, random_agent])
    
    payoffs_off_sum = 0.0
    
    # Simple Loop
    for _ in range(num_games):
         state, player_id = env.reset()
         while not env.is_over():
             action = env.agents[player_id].eval_step(state)
             if isinstance(action, tuple): action = action[0]
             state, player_id = env.step(action)
        
         # Extract Player 0 Payoff
         payoffs_off_sum += env.get_payoffs()[0]
         
    off_margin = payoffs_off_sum / num_games
    print(f"DEBUG: {env_name} Off Margin: {off_margin}")

    # 2. Random vs Agent (Agent is Defense/Player 1)
    # -----------------------------------------------
    env.set_agents([random_agent, agent])
    
    payoffs_def_sum = 0.0
    
    for _ in range(num_games):
         state, player_id = env.reset()
         while not env.is_over():
             action = env.agents[player_id].eval_step(state)
             if isinstance(action, tuple): action = action[0]
             state, player_id = env.step(action)
             
         # Extract Player 1 Payoff
         payoffs_def_sum += env.get_payoffs()[1]
         
    def_margin = payoffs_def_sum / num_games
    print(f"DEBUG: {env_name} Def Margin: {def_margin}")
    
    return off_margin, def_margin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', type=int, default=1000)
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    envs = [
        ('nfl-bucketed', "Standard (Perfect Info)"), 
        ('nfl-iig-bucketed', "IIG (Hidden Play Type)")
    ]
    
    agents = ['ppo', 'nfsp', 'deep_cfr']
    
    results = {}
    
    print(f"\n{'='*80}")
    print(f"{'Environment':<25} | {'Agent':<10} | {'Off Margin':<10} | {'Def Margin':<10} | {'Total':<10}")
    print(f"{'-'*80}")
    
    for env_name, env_label in envs:
        for agent_name in agents:
            # Load agent
            print(f"Loading {agent_name} for {env_name}...", end='\r')
            agent = load_agent(env_name, agent_name, device)
            
            if agent:
                off, def_ = evaluate_margin(agent, env_name, args.num_games)
                total = off + def_
                
                results[(env_name, agent_name)] = total
                
                print(f"{env_label:<25} | {agent_name.upper():<10} | {off:>10.3f} | {def_:>10.3f} | {total:>10.3f}")
            else:
                print(f"{env_label:<25} | {agent_name.upper():<10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")
                
    print(f"{'='*80}\n")
    
    # Value of Information Analysis
    print("Value of Information Analysis (Standard Margin - IIG Margin)")
    print("Positive = Value of knowing current play/formation vs pre-commitment")
    print(f"{'-'*60}")
    
    for agent_name in agents:
        std_score = results.get(('nfl-bucketed', agent_name))
        iig_score = results.get(('nfl-iig-bucketed', agent_name))
        
        if std_score is not None and iig_score is not None:
            delta = std_score - iig_score
            print(f"{agent_name.upper()}: {delta:+.3f} EPA")
        else:
            print(f"{agent_name.upper()}: Insufficient data")

if __name__ == '__main__':
    main()
