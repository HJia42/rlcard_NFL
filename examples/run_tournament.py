
import os
import torch
import numpy as np
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import set_seed

# Pre-generated scenarios for consistency
class Scenario:
    def __init__(self, seed):
        rng = np.random.RandomState(seed)
        self.formation = rng.choice(5) # 0-4
        self.play_type = rng.choice(2) # 0-1 (Pass/Run)
        self.box_count = rng.choice(5) # 0-4
        self.seed = seed

class ScenarioRandomAgent(object):
    '''Agent that replays pre-generated scenarios.'''
    def __init__(self, scenarios, role):
        self.scenarios = scenarios # List of Scenario objects
        self.role = role # 'offense' or 'defense'
        self.game_idx = 0
        self.use_raw = False

    def reset_counter(self):
        self.game_idx = 0

    def step(self, state):
        # Determine what phase we are in and return prescheduled action
        scenario = self.scenarios[self.game_idx]
        
        # Standard: Phase 0 (Form), Phase 1 (Box), Phase 2 (Play)
        # IIG: Phase 0 (Form), Phase 1 (Play), Phase 2 (Box)
        
        # But wait, step() is called multiple times per game.
        # How do we know which move we are making without consuming the index prematurely?
        # We need the environment to tell us the phase, which it does in state['phase'] (converted from int)
        # Or state['raw_obs'][3] is the phase int.
        
        phase = int(state['raw_obs'][3]) if 'raw_obs' in state else state['obs'][3]
        
        # Mapping based on verified file content:
        # Phase 0 is ALWAYS Formation (Offense)
        if phase == 0:
            return scenario.formation
            
        # Phase 1
        # In Standard: Defense picks Box (0-4)
        # In IIG: Offense picks Play (0-1)
        if phase == 1:
            if 'defense' in str(self.role): # If we are defense in Standard
                 return scenario.box_count
            else: # If we are offense in IIG
                 return scenario.play_type
                 
        # Phase 2
        # In Standard: Offense picks Play (0-1)
        # In IIG: Defense picks Box (0-4)
        if phase == 2:
             if 'offense' in str(self.role): # If we are offense in Standard
                 return scenario.play_type
             else: # If we are defense in IIG
                 return scenario.box_count
                 
        return 0 # Should not happen

    def eval_step(self, state):
        return self.step(state), {}

def _load_trained_agent(env, agent_type, env_name, device):
    """Internal helper to load trained agents (PPO, NFSP, DeepCFR)."""
    # Construct paths
    base_dir = f"experiments/{env_name}_{agent_type}"
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    # Sanitize State Shape
    # Checkpoint expects 12 inputs. env.state_shape might be [[12], [12]] which np.prod converts to 144.
    if isinstance(env.state_shape, list) and isinstance(env.state_shape[0], list):
        # Taking the first element's shape
        agent_state_shape = env.state_shape[0] 
    else:
        agent_state_shape = env.state_shape

    # Initialize Agent Structure
    if agent_type == 'ppo':
        agent = PPOAgent(
            num_actions=env.num_actions,
            state_shape=agent_state_shape,
            device=device
        )
        # Load latest checkpoint
        path = os.path.join(checkpoint_dir, "agent_0.pt")
        if os.path.exists(path):
            agent.load(path)
        else:
            print(f"  [Warning] PPO model not found at {path}")
            return None

    elif agent_type == 'nfsp':
        agent = NFSPAgent(
            num_actions=env.num_actions,
            state_shape=agent_state_shape,
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

def load_agent_with_scenarios(env, agent_type, env_name, device, scenarios, role):
    """Load agent, injecting scenarios if random."""
    if agent_type == 'random':
        return ScenarioRandomAgent(scenarios, role)
    
    return _load_trained_agent(env, agent_type, env_name, device)

def run_head_to_head(env, offense_agent, defense_agent, scenarios):
    """Run matchups using shared scenarios."""
    env.set_agents([offense_agent, defense_agent])
    total_epa = 0.0
    
    # Reset scenario counters if they are ScenarioAgents
    if hasattr(offense_agent, 'reset_counter'): offense_agent.reset_counter()
    if hasattr(defense_agent, 'reset_counter'): defense_agent.reset_counter()
    
    for i in range(len(scenarios)):
        # Increment counters manually at start of game loop (since step calls logic is stateless)
        # Actually ScenarioAgent needs to know current game index
        if hasattr(offense_agent, 'game_idx'): offense_agent.game_idx = i
        if hasattr(defense_agent, 'game_idx'): defense_agent.game_idx = i
        
        # We should also seed the environment for outcome generation consistency
        env.seed(scenarios[i].seed)
        
        state, player_id = env.reset()
        while not env.is_over():
            action = env.agents[player_id].eval_step(state)
            if isinstance(action, tuple): action = action[0]
            state, player_id = env.step(action)
        
        total_epa += env.get_payoffs()[0]
        
    return total_epa / len(scenarios)

# Define Scenarios Global
NUM_GAMES = 5000
GLOBAL_SCENARIOS = [Scenario(seed=42+i) for i in range(NUM_GAMES)]

# Updated run_league signature to use scenarios
def run_league(league_name, env_name, agent_names):
    print(f"\n{'='*60}")
    print(f"Starting {league_name} (Env: {env_name})")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Initialize Environment
    if 'iig' in env_name:
        from rlcard.envs.nfl_iig import NFLIIGEnv
        env = NFLIIGEnv(config={'allow_step_back':False, 'seed':42})
    else:
        from rlcard.envs.nfl import NFLEnv
        env = NFLEnv(config={'allow_step_back':False, 'seed':42})
        
    # Enable Cached Model for Speed
    env.game.use_cached_model = True
    env.game.use_simple_model = False
            
    # 2. Round Robin
    matrix = np.zeros((len(agent_names), len(agent_names)))
    
    print("\nRunning Matchups (Scenario Based)...")
    header = f"{'Off / Def':<12} | " + " | ".join([f"{name:^10}" for name in agent_names])
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i, off_name in enumerate(agent_names):
        row_str = f"{off_name:<12} | "
        for j, def_name in enumerate(agent_names):
            # Load Agents FRESH for every matchup to ensure clean state
            # Pass Role for Random Agent scenario interpretation
            off_agent = load_agent_with_scenarios(env, off_name.lower(), env_name, device, GLOBAL_SCENARIOS, role='offense')
            def_agent = load_agent_with_scenarios(env, def_name.lower(), env_name, device, GLOBAL_SCENARIOS, role='defense')
            
            if off_agent is None or def_agent is None:
                matrix[i, j] = np.nan
                row_str += f"{'N/A':^10} | "
                continue

            # Set Eval Mode
            if hasattr(off_agent, 'set_mode'): off_agent.set_mode(rlcard.agents.Agent.EVAL)
            if hasattr(def_agent, 'set_mode'): def_agent.set_mode(rlcard.agents.Agent.EVAL)
            
            avg_epa = run_head_to_head(env, off_agent, def_agent, GLOBAL_SCENARIOS)
            matrix[i, j] = avg_epa
            row_str += f"{avg_epa:^10.3f} | "
            
        print(row_str)
        
    return matrix

if __name__ == '__main__':
    # Define Agents
    agent_list = ['PPO', 'NFSP', 'Deep_CFR', 'Random']
    
    # Run Standard League
    run_league("Standard League (Perfect Info)", "nfl-bucketed", agent_list)
    
    # Run IIG League
    run_league("IIG League (Hidden Info)", "nfl-iig-bucketed", agent_list)

