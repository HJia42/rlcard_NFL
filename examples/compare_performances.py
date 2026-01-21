
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

def evaluate_margin(agent, env_name, num_games, num_iterations):
    """
    Calculate EPA Margin against Standard Random Baseline with Confidence Intervals.
    Returns: 
        off_margin: Mean Offense Margin
        off_ci: 95% CI for Offense
        def_margin: Mean Defense Margin
        def_ci: 95% CI for Defense
    """
    env = rlcard.make(env_name, config={'single_play': True})
    
    # Lists to store mean results from each iteration
    off_means = []
    def_means = []
    
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    # Run N Iterations
    for it in range(num_iterations):
        # 1. Agent vs Random (Agent is Offense/Player 0)
        # -----------------------------------------------
        env.set_agents([agent, random_agent])
        payoffs_off_sum = 0.0
        
        for _ in range(num_games):
             state, player_id = env.reset()
             while not env.is_over():
                 action = env.agents[player_id].eval_step(state)
                 if isinstance(action, tuple): action = action[0]
                 state, player_id = env.step(action)
            
             payoffs_off_sum += env.get_payoffs()[0]
             
        off_means.append(payoffs_off_sum / num_games)
        
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
                 
             payoffs_def_sum += env.get_payoffs()[1]
             
        def_means.append(payoffs_def_sum / num_games)
        print(f"  Iteration {it+1}/{num_iterations} complete...", end='\r')

    # Calculate Statistics
    off_mean = np.mean(off_means)
    off_std = np.std(off_means, ddof=1) # Sample STD
    off_ci = 1.96 * (off_std / np.sqrt(num_iterations)) # 95% CI
    
    def_mean = np.mean(def_means)
    def_std = np.std(def_means, ddof=1)
    def_ci = 1.96 * (def_std / np.sqrt(num_iterations))
    
    print(f"  Completed {num_iterations} iterations.                              ")
    print(f"DEBUG: {env_name} Off: {off_mean:.3f} +/- {off_ci:.3f}")
    print(f"DEBUG: {env_name} Def: {def_mean:.3f} +/- {def_ci:.3f}")
    
    return off_mean, off_ci, def_mean, def_ci

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_games', type=int, default=10000, help="Games per iteration")
    parser.add_argument('--num_iterations', type=int, default=100, help="Number of iterations for CI")
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Config: {args.num_iterations} iterations of {args.num_games} games each.")
    
    envs = [
        ('nfl-bucketed', "Standard (Perfect Info)"), 
        ('nfl-iig-bucketed', "IIG (Hidden Play Type)")
    ]
    
    agents = ['ppo', 'nfsp', 'deep_cfr']
    
    results = {}
    
    print(f"\n{'='*100}")
    print(f"{'Environment':<25} | {'Agent':<10} | {'Off Margin (95% CI)':<20} | {'Def Margin (95% CI)':<20} | {'Total':<10}")
    print(f"{'-'*100}")
    
    for env_name, env_label in envs:
        for agent_name in agents:
            # Load agent
            print(f"Loading {agent_name} for {env_name}...", end='\r')
            agent = load_agent(env_name, agent_name, device)
            
            if agent:
                off_m, off_ci, def_m, def_ci = evaluate_margin(agent, env_name, args.num_games, args.num_iterations)
                total = off_m + def_m
                
                # Store Mean and CI for both Offense and Defense for later analysis if needed
                results[(env_name, agent_name)] = {'total': total, 'off': off_m, 'def': def_m}
                
                off_str = f"{off_m:.3f} +/- {off_ci:.3f}"
                def_str = f"{def_m:.3f} +/- {def_ci:.3f}"
                
                print(f"{env_label:<25} | {agent_name.upper():<10} | {off_str:>20} | {def_str:>20} | {total:>10.3f}")
            else:
                print(f"{env_label:<25} | {agent_name.upper():<10} | {'N/A':>20} | {'N/A':>20} | {'N/A':>10}")
                
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
