
import os
import torch
import numpy as np
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import set_seed

class SeededRandomAgent(object):
    '''Random agent with fixed seed for fair comparison.'''
    def __init__(self, num_actions, seed=None):
        self.num_actions = num_actions
        self.use_raw = False
        self.rng = np.random.RandomState(seed) if seed is not None else np.random

    def step(self, state):
        return self.rng.choice(list(state['legal_actions'].keys()))

    def eval_step(self, state):
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])
        info = {}
        return self.step(state), info

def load_agent(env, agent_type, env_name, device, seed=42):
    """Load a trained agent or return evaluation baseline."""
    if agent_type == 'random':
        return SeededRandomAgent(num_actions=env.num_actions, seed=seed)
    
    # Construct paths
    base_dir = f"experiments/{env_name}_{agent_type}"
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    # Initialize Agent Structure
    if agent_type == 'ppo':
        agent = PPOAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            device=device
        )
        # Load latest checkpoint
        path = os.path.join(checkpoint_dir, "agent_0.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            agent.load(checkpoint)
        else:
            print(f"  [Warning] PPO model not found at {path}")
            return None

    elif agent_type == 'nfsp':
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=env.state_shape,
            hidden_layers_sizes=[64, 64],
            q_mlp_layers=[64, 64],
            device=device
        )
        path = os.path.join(checkpoint_dir, "agent_0.pt") # Checkpoint folder
        if os.path.exists(path):
            agent.load(path)
        else:
             print(f"  [Warning] NFSP model not found at {path}")
             return None

    elif agent_type == 'deep_cfr':
        agent = DeepCFRAgent(
            env,
            device=device
        )
        # DeepCFR Path Logic (Fixed)
        path = os.path.join(checkpoint_dir, 'model.pt')
        if not os.path.exists(path):
            path = os.path.join(base_dir, 'model.pt')
        
        if os.path.exists(path):
            agent.model_path = os.path.dirname(path)
            if not agent.load():
                 print(f"  [Error] DeepCFR internal load failed for {path}")
                 return None
        else:
            print(f"  [Warning] DeepCFR model not found at {path}")
            return None
    
    else:
        return None
        
    return agent

def run_head_to_head(env, offense_agent, defense_agent, num_games=2000):
    """Run a specific matchups and return average Payoff (EPA)."""
    # Set agents
    env.set_agents([offense_agent, defense_agent])
    
    total_epa = 0.0
    
    for _ in range(num_games):
        state, player_id = env.init_game()
        while not env.is_over():
            action = env.agents[player_id].eval_step(state) # Use eval_step
            # Handle different return formats (DeepCFR returns int, PPO returns tuple)
            if isinstance(action, tuple):
                 action = action[0]
            
            state, player_id = env.step(action)
        
        # Game over - get offense payoff (index 0)
        total_epa += env.get_payoffs()[0]
        
    return total_epa / num_games

def run_league(league_name, env_name, agent_names, num_games=2000):
    print(f"\n{'='*60}")
    print(f"Starting {league_name} (Env: {env_name})")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Environment
    # We create a dummy env just to initialize agents
    if 'iig' in env_name:
        from rlcard.envs.nfl_iig import NFLIIGEnv
        env = NFLIIGEnv(config={'allow_step_back':False, 'seed':42}) # Raw env for now/wrapper later
        # Actually use make to get correct wrappers if needed, but wrapper often changes state shape
        # Let's use raw class to avoid registration issues for now, or use make if registered
        # Re-using direct class instantiation for safety as in train script
    else:
        from rlcard.envs.nfl import NFLEnv
        env = NFLEnv(config={'allow_step_back':False, 'seed':42})
        
    # Enable Cached Model for Speed
    env.game.use_cached_model = True
    env.game.use_simple_model = False
    
    # 2. Load All Agents
    agents = {}
    for name in agent_names:
        print(f"Loading {name}...")
        agent = load_agent(env, name.lower(), env_name, device)
        if agent is None:
            print(f"  FAILED to load {name}")
        else:
            # Set to eval mode if applicable
            if hasattr(agent, 'set_mode'):
                agent.set_mode(rlcard.agents.Agent.EVAL)
            agents[name] = agent
            
    # 3. Round Robin
    # Row = Offense, Col = Defense
    matrix = np.zeros((len(agent_names), len(agent_names)))
    
    print("\nRunning Matchups...")
    header = f"{'Off / Def':<12} | " + " | ".join([f"{name:^10}" for name in agent_names])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i, off_name in enumerate(agent_names):
        row_str = f"{off_name:<12} | "
        for j, def_name in enumerate(agent_names):
            if off_name not in agents or def_name not in agents:
                matrix[i, j] = np.nan
                row_str += f"{'N/A':^10} | "
                continue
                
            off_agent = agents[off_name]
            def_agent = agents[def_name]
            
            avg_epa = run_head_to_head(env, off_agent, def_agent, num_games)
            matrix[i, j] = avg_epa
            row_str += f"{avg_epa:^10.3f} | "
            
        print(row_str)
        
    return matrix

if __name__ == '__main__':
    # Define Agents
    agent_list = ['PPO', 'NFSP', 'Deep_CFR', 'Random']
    
    # Run Standard League
    run_league("Standard League (Perfect Info)", "nfl-bucketed", agent_list, num_games=5000)
    
    # Run IIG League
    run_league("IIG League (Hidden Info)", "nfl-iig-bucketed", agent_list, num_games=5000)
