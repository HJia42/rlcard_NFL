"""
Unified Agent Loader for NFL RL

Load any trained agent type with a single interface.
Consolidates loading logic from compare_agents.py.
"""

import os
from typing import Optional, Tuple, Any

import torch
import rlcard


SUPPORTED_AGENT_TYPES = ['ppo', 'dmc', 'nfsp', 'cfr', 'mccfr', 'deep_cfr']


def load_agent(
    agent_type: str,
    model_path: str,
    game: str = 'nfl-bucketed',
    verbose: bool = True,
) -> Tuple[Optional[Any], Any]:
    """
    Load a trained agent from disk.
    
    Args:
        agent_type: One of 'ppo', 'dmc', 'nfsp', 'cfr', 'mccfr', 'deep_cfr'
        model_path: Path to model file or directory
        game: Environment name ('nfl' or 'nfl-bucketed')
        verbose: Print loading status
    
    Returns:
        Tuple of (agent, environment) or (None, None) on failure
    """
    agent_type = agent_type.lower()
    
    if agent_type not in SUPPORTED_AGENT_TYPES:
        raise ValueError(f"Unsupported agent type: {agent_type}. "
                        f"Supported: {SUPPORTED_AGENT_TYPES}")
    
    # Create environment
    env_config = {'single_play': True, 'use_cached_model': True}
    
    if agent_type in ['cfr', 'mccfr']:
        env_config['allow_step_back'] = True
    
    env = rlcard.make(game, config=env_config)
    
    try:
        if agent_type == 'ppo':
            agent = _load_ppo(model_path, env, verbose)
        elif agent_type == 'dmc':
            agent = _load_dmc(model_path, env, verbose)
        elif agent_type == 'nfsp':
            agent = _load_nfsp(model_path, env, verbose)
        elif agent_type == 'cfr':
            agent = _load_cfr(model_path, env, verbose)
        elif agent_type == 'mccfr':
            agent = _load_mccfr(model_path, env, verbose)
        elif agent_type == 'deep_cfr':
            agent = _load_deep_cfr(model_path, env, verbose)
        else:
            return None, None
        
        return agent, env
    
    except Exception as e:
        if verbose:
            print(f"Failed to load {agent_type} agent from {model_path}: {e}")
        return None, None


def _load_ppo(model_path: str, env, verbose: bool) -> Any:
    """Load PPO agent."""
    from rlcard.agents.ppo_agent import PPOAgent
    
    # If model_path is a directory, find the final .pt file
    if os.path.isdir(model_path):
        pt_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
        # Prefer 'final' checkpoint
        final_files = [f for f in pt_files if 'final' in f]
        if final_files:
            model_path = os.path.join(model_path, final_files[0])
        elif pt_files:
            # Use the highest numbered checkpoint
            pt_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'), reverse=True)
            model_path = os.path.join(model_path, pt_files[0])
        else:
            raise FileNotFoundError(f"No .pt files found in {model_path}")
    
    # First peek at the checkpoint to get the actual state shape and action count
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Infer state shape from first layer weights: shared.0.weight has shape [hidden, state_dim]
    state_dict = checkpoint.get('network_state_dict', checkpoint)
    if 'shared.0.weight' in state_dict:
        state_dim = state_dict['shared.0.weight'].shape[1]
    else:
        # Fallback to environment
        state_dim = env.state_shape[0][0] if isinstance(env.state_shape[0], list) else env.state_shape[0]
    
    # Infer num_actions from actor head: actor_head.2.weight has shape [num_actions, hidden]
    if 'actor_head.2.weight' in state_dict:
        num_actions = state_dict['actor_head.2.weight'].shape[0]
    else:
        # Fallback to environment
        num_actions = env.num_actions
    
    agent = PPOAgent(
        state_shape=[state_dim],
        num_actions=num_actions,
        hidden_dims=[128, 128],
    )
    agent.load(model_path)
    
    if verbose:
        print(f"Loaded PPO agent from {model_path} (state_dim={state_dim}, num_actions={num_actions})")
    
    return agent


def _load_dmc(model_path: str, env, verbose: bool) -> Any:
    """Load DMC agent."""
    from rlcard.agents.dmc_agent.model import DMCAgent
    
    model_file = os.path.join(model_path, 'model.tar')
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"DMC model not found at {model_file}")
    
    checkpoint = torch.load(model_file, map_location='cpu')
    model_state_dicts = checkpoint['model_state_dict']
    
    # Load player 0 agent (offense)
    state_shape = env.state_shape[0]
    action_shape = (env.num_actions,)
    
    agent = DMCAgent(
        state_shape=state_shape,
        action_shape=action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        device='cpu'
    )
    agent.net.load_state_dict(model_state_dicts[0])
    agent.net.eval()
    
    if verbose:
        frames = checkpoint.get('frames', 0)
        print(f"Loaded DMC agent from {model_path} ({int(frames):,} frames)")
    
    return agent


def _load_nfsp(model_path: str, env, verbose: bool) -> Any:
    """Load NFSP agent."""
    from rlcard.agents import NFSPAgent
    
    # model_path can be a directory or a .pt file
    if os.path.isdir(model_path):
        # Search for checkpoint file - try multiple patterns
        game_name = getattr(env, 'name', getattr(env, 'game_name', 'nfl-bucketed'))
        patterns = [
            f'nfsp_{game_name}_p0_final.pt',
            'nfsp_nfl-bucketed_p0_final.pt',
            'nfsp_nfl_p0_final.pt',
            'nfsp_player_0_final.pt',
        ]
        checkpoint_path = None
        for pattern in patterns:
            path = os.path.join(model_path, pattern)
            if os.path.exists(path):
                checkpoint_path = path
                break
        
        if checkpoint_path is None:
            raise FileNotFoundError(f"No NFSP checkpoint found in {model_path}")
    else:
        checkpoint_path = model_path
    
    # Load the checkpoint dictionary first
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Use from_checkpoint method
    agent = NFSPAgent.from_checkpoint(checkpoint)
    
    if verbose:
        print(f"Loaded NFSP agent from {checkpoint_path}")
    
    return agent


def _load_cfr(model_path: str, env, verbose: bool) -> Any:
    """Load tabular CFR agent."""
    from rlcard.agents import CFRAgent
    
    agent = CFRAgent(env, model_path)
    agent.load()
    
    if verbose:
        print(f"Loaded CFR agent from {model_path}")
    
    return agent


def _load_mccfr(model_path: str, env, verbose: bool) -> Any:
    """Load MCCFR agent."""
    from rlcard.agents.mccfr_agent import MCCFRAgent
    
    agent = MCCFRAgent(env, model_path)
    agent.load()
    
    if verbose:
        print(f"Loaded MCCFR agent from {model_path}")
    
    return agent


def _load_deep_cfr(model_path: str, env, verbose: bool) -> Any:
    """Load Deep CFR agent via ParallelDeepCFRTrainer."""
    from rlcard.agents.parallel_deep_cfr import ParallelDeepCFRTrainer
    
    # Get game name - IIG envs use 'name', standard envs use 'game_name'
    game_name = getattr(env, 'name', getattr(env, 'game_name', 'nfl-bucketed'))
    
    # Deep CFR requires loading through the trainer
    trainer = ParallelDeepCFRTrainer(
        env_config={'game': game_name, 'single_play': True},
        model_path=model_path,
    )
    
    if not trainer.load():
        raise FileNotFoundError(f"Could not load Deep CFR model from {model_path}")
    
    agent = trainer.get_agent()
    
    if verbose:
        print(f"Loaded Deep CFR agent from {model_path}")
    
    return agent


def list_available_models(base_dir: str = 'models') -> dict:
    """
    Scan for available trained models.
    
    Returns:
        Dict mapping agent_type -> list of model paths
    """
    models = {t: [] for t in SUPPORTED_AGENT_TYPES}
    
    if not os.path.exists(base_dir):
        return models
    
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        
        # Check for PPO models
        if 'ppo' in item.lower() and item.endswith('.pt'):
            models['ppo'].append(item_path)
        
        # Check for NFSP models
        if 'nfsp' in item.lower() and item.endswith('.pt'):
            models['nfsp'].append(item_path)
        
        # Check for DMC directories
        if os.path.isdir(item_path) and 'dmc' in item.lower():
            if os.path.exists(os.path.join(item_path, 'model.tar')):
                models['dmc'].append(item_path)
        
        # Check for CFR directories
        if os.path.isdir(item_path) and 'cfr' in item.lower():
            if 'deep' in item.lower():
                models['deep_cfr'].append(item_path)
            elif 'mccfr' in item.lower():
                models['mccfr'].append(item_path)
            else:
                models['cfr'].append(item_path)
    
    return models
