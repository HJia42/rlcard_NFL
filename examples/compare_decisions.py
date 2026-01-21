import os
import torch
import numpy as np
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
import pandas as pd

def load_agent(env_name, agent_type, device='cpu'):
    """Load a trained agent helper."""
    env = rlcard.make(env_name, config={'single_play': True})
    if isinstance(env.state_shape, list) and isinstance(env.state_shape[0], list):
         agent_state_shape = env.state_shape[0]
    else:
         agent_state_shape = env.state_shape
    
    base_dir = f'experiments/{env_name}_{agent_type}'
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    
    try:
        if agent_type == 'ppo':
            agent = PPOAgent(state_shape=agent_state_shape, num_actions=env.num_actions, hidden_dims=[128, 128], device=device)
            agent.load(os.path.join(checkpoint_dir, 'agent_0.pt'))
        elif agent_type == 'nfsp':
            agent = NFSPAgent(num_actions=env.num_actions, state_shape=agent_state_shape, hidden_layers_sizes=[128, 128], q_mlp_layers=[128, 128], device=device)
            agent.load(os.path.join(checkpoint_dir, 'agent_0.pt'))
        elif agent_type == 'deep_cfr':
            agent = DeepCFRAgent(env, device=device)
            path = os.path.join(checkpoint_dir, 'model.pt')
            agent.model_path = os.path.dirname(path) if os.path.exists(path) else base_dir
            if not agent.load(): return None
        else:
            return None
        return agent
    except Exception as e:
        print(f"Error loading {agent_type}: {e}")
        return None

def get_action_prob(agent, state):
    """Get probability distribution of actions for a given state."""
    # This varies by agent API
        
    # Manual extraction based on agent type
    if isinstance(agent, PPOAgent):
        # PPO has internal `network` (ActorCritic) not `policy`
        obs = state['obs']
        state_tensor = torch.from_numpy(obs).float().to(agent.device).unsqueeze(0)
        with torch.no_grad():
             # ActorCritic forward pass manual reconstruction
             features = agent.network.shared(state_tensor)
             logits = agent.network.actor(features)
             probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs
        
    elif isinstance(agent, NFSPAgent):
        # NFSP uses DQN policy for evaluation usually
        obs = state['obs']
        q_values = agent.q_estimator.predict_nograd(np.expand_dims(obs, 0))[0]
        # Softmax Q values to get a "Policy" view
        probs = np.exp(q_values) / np.sum(np.exp(q_values))
        return probs
        
    elif isinstance(agent, DeepCFRAgent):
        # DeepCFR returns policy directly
        action, info = agent.eval_step(state)
        if 'probs' in info:
            # Info['probs'] is the strategy
             # DeepCFR output is keyed by action index, we need valid list
             full_probs = np.zeros(agent.num_actions)
             for a, p in info['probs'].items():
                 full_probs[a] = p
             return full_probs
    
    return np.zeros(2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define interesting States (Formations)
    # We fix Down/Distance to standard 1st & 10 to isolate Formation effect
    # Formation Indices: 0: Shotgun, 1: Singleback, 2: I_Form, 3: Pistol, 4: Empty (Example mapping)
    formations = {
        0: 'Shotgun',
        1: 'Singleback',
        2: 'I_Form',
        3: 'Pistol', 
        4: 'Empty'
    }
    
    agents = ['ppo', 'deep_cfr'] # NFSP output is Q-values, less comparable directly to prob
    
    print(f"{'Agent':<10} | {'Formation':<15} | {'Std Pass%':<10} | {'IIG Pass%':<10} | {'Delta (Std-IIG)':<15}")
    print("-" * 75)

    for agent_name in agents:
        # Load Both Versions
        std_agent = load_agent('nfl-bucketed', agent_name, device)
        iig_agent = load_agent('nfl-iig-bucketed', agent_name, device)
        if not std_agent or not iig_agent: continue
        
        for f_idx, f_name in formations.items():
            # Construct a Synthetic State
            # Vector: [Down, Ytg, Yardline, Phase, ...FormationOneHot...]
            # NOTE: Different environments have different vector constructions!
            # We must use the REAL environment to generate the state to be safe.
            
            # Helper to generate state
            def get_state(env_name, formation_idx):
                env = rlcard.make(env_name, config={'single_play': True})
                env.reset()
                # Hack: Force the state to match our criteria
                # We need to simulate being in the Phase where Offense chooses Play Type
                # Std: Phase 2 (After Def Box). IIG: Phase 1 (Before Def Box).
                
                # ... This is tricky because Std Agent expects Box Count in state, IIG does not.
                # If we compare them, we must give Std Agent a "Average" box or iterate all boxes?
                # Let's assume Box Count 2 (Neutral/Nickel) for Std Agent comparison.
                
                env.game.down = 1
                env.game.ydstogo = 10
                env.game.yardline = 25
                env.game.pending_formation = formation_idx # Set formation
                env.game.current_player = 0
                
                if 'iig' in env_name:
                    env.game.phase = 1 # Play Type Selection
                    state = env.get_state(0) # Player 0
                else: 
                    # Standard
                    env.game.phase = 2 # Play Type Selection
                    # Defense Action must be valid (Box, Personnel) tuple from DEFENSE_ACTIONS
                    # Box 6 is neutral-ish. Personnel is 'Standard'.
                    env.game.pending_defense_action = (6, 'Standard') 
                    state = env.get_state(0)
                    
                return state
            
            std_state = get_state('nfl-bucketed', f_idx)
            iig_state = get_state('nfl-iig-bucketed', f_idx)
            
            # Predict
            # Actions: 0=Pass, 1=Run (Check specific mapping in game.py PLAY_TYPE_ACTIONS)
            # Actually usually 0=Pass, 1=Run.
            
            std_probs = get_action_prob(std_agent, std_state)
            iig_probs = get_action_prob(iig_agent, iig_state)
            
            # Check dimensions. If Q-values, normalize or just print preference?
            # DeepCFR/PPO return probs.
            # Assuming Action 0 is Pass. 
            std_pass = std_probs[0]
            iig_pass = iig_probs[0]
            
            delta = std_pass - iig_pass
            
            print(f"{agent_name.upper():<10} | {f_name:<15} | {std_pass:.3f}      | {iig_pass:.3f}      | {delta:+.3f}")
            
if __name__ == '__main__':
    main()
