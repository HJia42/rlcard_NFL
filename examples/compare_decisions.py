import os
import torch
import numpy as np
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent

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
            # PPO Load with weights_only=False fix if needed (default in older torch is fine warning)
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
        # print(f"Error loading {agent_type}: {e}")
        return None

def get_action_prob(agent, state):
    """Get probability distribution of actions for a given state."""
    if isinstance(agent, PPOAgent):
        # PPO has internal `network` (ActorCritic)
        obs = state['obs']
        state_tensor = torch.from_numpy(obs).float().to(agent.device).unsqueeze(0)
        with torch.no_grad():
             features = agent.network.shared(state_tensor)
             logits = agent.network.actor(features)
             probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs
        
    elif isinstance(agent, DeepCFRAgent):
        action, info = agent.eval_step(state)
        if 'probs' in info:
             full_probs = np.zeros(agent.num_actions)
             str_map = {'pass': 0, 'rush': 1}
             for a, p in info['probs'].items():
                 idx = -1
                 if isinstance(a, str):
                     if a in str_map: idx = str_map[a]
                     else:
                         try: idx = int(a)
                         except: pass
                 else:
                     idx = int(a)
                 if idx >= 0 and idx < len(full_probs):
                     full_probs[idx] = p
             return full_probs
    
    return np.zeros(2)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define interesting Situations (Down, Distance)
    # We fix Formation to SHOTGUN (Neutral/Passing baseline) 
    # and Yardline to 40 (Mid-field, avoiding Redzone edge cases initially)
    situations = [
        (1, 10, "1st & 10"),
        (2, 10, "2nd & Long (10)"),
        (2, 5,  "2nd & Med (5)"),
        (2, 1,  "2nd & Short (1)"),
        (3, 10, "3rd & Long (10)"),
        (3, 5,  "3rd & Med (5)"),
        (3, 1,  "3rd & Short (1)"),
        (4, 1,  "4th & Short (1)"),
        (4, 5,  "4th & Med (5)")
    ]
    
    fixed_formation = "SHOTGUN"
    fixed_yardline = 40
    
    agents = ['ppo', 'deep_cfr'] 
    
    print(f"\nComparing Decision Policies (Fixed Formation: {fixed_formation})")
    print(f"{'Agent':<10} | {'Situation':<20} | {'Std Pass%':<10} | {'IIG Pass%':<10} | {'Delta':<10}")
    print("-" * 75)

    for agent_name in agents:
        std_agent = load_agent('nfl-bucketed', agent_name, device)
        iig_agent = load_agent('nfl-iig-bucketed', agent_name, device)
        if not std_agent or not iig_agent: continue
        
        results = []
        
        for down, dist, label in situations:
            def get_state(env_name, d, dst):
                env = rlcard.make(env_name, config={'single_play': True})
                env.reset()
                env.game.down = d
                env.game.ydstogo = dst
                env.game.yardline = fixed_yardline
                env.game.pending_formation = fixed_formation 
                env.game.current_player = 0
                
                if 'iig' in env_name:
                    env.game.phase = 1 
                    state = env.get_state(0)
                else: 
                    env.game.phase = 2 
                    # Neutral Box for Standard
                    env.game.pending_defense_action = (6, 'Standard') 
                    state = env.get_state(0)
                return state
            
            std_state = get_state('nfl-bucketed', down, dist)
            iig_state = get_state('nfl-iig-bucketed', down, dist)
            
            std_probs = get_action_prob(std_agent, std_state)
            iig_probs = get_action_prob(iig_agent, iig_state)
            
            # Action 0 = Pass, 1 = Run
            std_pass = std_probs[0]
            iig_pass = iig_probs[0]
            delta = std_pass - iig_pass
            
            results.append((label, std_pass, iig_pass, delta))
            
        # Sort by absolute delta to find biggest divergence
        results.sort(key=lambda x: abs(x[3]), reverse=True)
        
        for label, sp, ip, d in results:
             print(f"{agent_name.upper():<10} | {label:<20} | {sp:.3f}      | {ip:.3f}      | {d:+.3f}")
        print("-" * 75)
            
if __name__ == '__main__':
    main()
