import os
import torch
import numpy as np
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent

import sys

class Quiet:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def load_agent(env_name, agent_type, device='cpu'):
    """Load a trained agent helper."""
    # We can rely on Quiet in main() covering this, or wrap here too.
    # Let's clean up this function to not create env if not needed or just be quiet.
    with Quiet():
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
            with Quiet():
                agent.load(os.path.join(checkpoint_dir, 'agent_0.pt'))
        elif agent_type == 'deep_cfr':
            agent = DeepCFRAgent(env, device=device)
            path = os.path.join(checkpoint_dir, 'model.pt')
            agent.model_path = os.path.dirname(path) if os.path.exists(path) else base_dir
            with Quiet():
                if not agent.load(): return None
        else:
            return None
        return agent
    except Exception:
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

    # Pre-load environments ONCE to avoid repeated "Loaded..." logs
    with Quiet():
        std_env_instance = rlcard.make('nfl-bucketed', config={'single_play': True})
        iig_env_instance = rlcard.make('nfl-iig-bucketed', config={'single_play': True})

    for agent_name in agents:
        std_agent = load_agent('nfl-bucketed', agent_name, device)
        iig_agent = load_agent('nfl-iig-bucketed', agent_name, device)
        if not std_agent or not iig_agent: continue
        
        results = []
        
        for down, dist, label in situations:
            def get_state(env, d, dst):
                env.reset()
                env.game.down = d
                env.game.ydstogo = dst
                env.game.yardline = fixed_yardline
                env.game.pending_formation = fixed_formation 
                env.game.current_player = 0
                
                # Check based on env properties or name if we had it. 
                # Better to just use the instance logic
                if isinstance(env.game, rlcard.games.nfl.game_iig.NFLGameIIG): # Safe instance check if imported
                    env.game.phase = 1 
                # Or just duck typing/attribute check
                elif hasattr(env.game, 'committed_play_type'):
                     env.game.phase = 1
                else: 
                     env.game.phase = 2 
                     env.game.pending_defense_action = (6, 'Standard') 
                
                # Actually, simpliest way since we pass explicit envs:
                # We know std_env is std, iig is iig.
                # Just logic based on caller.
                return env.get_state(0)

            # Manual setup for known env types
            # Std Env
            std_env_instance.reset()
            std_env_instance.game.down = down
            std_env_instance.game.ydstogo = dist
            std_env_instance.game.yardline = fixed_yardline
            std_env_instance.game.pending_formation = fixed_formation
            std_env_instance.game.current_player = 0
            std_env_instance.game.phase = 2
            std_env_instance.game.pending_defense_action = (6, 'Standard')
            std_state = std_env_instance.get_state(0)
            
            # IIG Env
            iig_env_instance.reset()
            iig_env_instance.game.down = down
            iig_env_instance.game.ydstogo = dist
            iig_env_instance.game.yardline = fixed_yardline
            iig_env_instance.game.pending_formation = fixed_formation
            iig_env_instance.game.current_player = 0
            iig_env_instance.game.phase = 1 
            iig_state = iig_env_instance.get_state(0)
            
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
