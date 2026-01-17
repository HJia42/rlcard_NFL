"""
Agent Round Robin Tournament

Pits PPO, DMC, and NFSP agents against each other in a round-robin tournament.
Each agent plays as both offense and defense against all other agents.
"""

import argparse
import os
import sys
sys.path.insert(0, '.')

import torch
import numpy as np
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.dmc_agent.model import DMCAgent


def load_ppo_agent(model_path, env):
    """Load PPO agent."""
    agent = PPOAgent(
        state_shape=env.state_shape[0], 
        num_actions=7, 
        hidden_dims=[128, 128]
    )
    agent.load(model_path)
    return agent


def load_nfsp_agent(model_path, env):
    """Load NFSP agent."""
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    agent = NFSPAgent.from_checkpoint(checkpoint)
    return agent


def load_dmc_agent(model_dir, env, player_id=0):
    """Load DMC agent for a specific player."""
    model_file = os.path.join(model_dir, 'model.tar')
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    
    model_state_dicts = checkpoint['model_state_dict']
    state_shape = env.state_shape[player_id]
    action_shape = (env.num_actions,)
    
    agent = DMCAgent(
        state_shape=state_shape,
        action_shape=action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        device='cpu'
    )
    agent.net.load_state_dict(model_state_dicts[player_id])
    agent.net.eval()
    return agent


class RandomAgent:
    """Random baseline agent with standardized eval_step output."""
    def __init__(self, num_actions):
        self.num_actions = num_actions
    
    def eval_step(self, state):
        legal_actions = state.get('legal_actions', list(range(self.num_actions)))
        if isinstance(legal_actions, dict):
            legal_actions = list(legal_actions.keys())
        action = np.random.choice(legal_actions) if legal_actions else 0
        
        # Return standardized probs format with str keys
        raw_actions = state.get('raw_legal_actions', legal_actions)
        probs = {raw_actions[i]: 1.0/len(legal_actions) for i in range(len(legal_actions))}
        return action, {'probs': probs}


def get_action_from_agent(agent, agent_type, state, env):
    """Get action from agent - all agents now use uniform eval_step() interface.
    
    Args:
        agent: Agent instance
        agent_type: (deprecated) Agent type string - no longer needed
        state: State dictionary from environment
        env: Environment (for compatibility, not used)
    
    Returns:
        action: Selected action index
    """
    # All agents now use the same eval_step() interface
    action, _ = agent.eval_step(state)
    return action


def run_games(agents, agent_types, env, num_games=100):
    """Run games between two agents."""
    total_rewards = [0.0, 0.0]
    wins = [0, 0, 0]  # [offense wins, defense wins, draws]
    
    for _ in range(num_games):
        state, player_id = env.reset()
        
        while not env.is_over():
            agent = agents[player_id]
            agent_type = agent_types[player_id]
            action = get_action_from_agent(agent, agent_type, state, env)
            state, player_id = env.step(action)
        
        payoffs = env.get_payoffs()
        total_rewards[0] += payoffs[0]
        total_rewards[1] += payoffs[1]
        
        if payoffs[0] > payoffs[1]:
            wins[0] += 1  # Offense wins
        elif payoffs[1] > payoffs[0]:
            wins[1] += 1  # Defense wins
        else:
            wins[2] += 1  # Draw
    
    return {
        'avg_epa_offense': total_rewards[0] / num_games,
        'avg_epa_defense': total_rewards[1] / num_games,
        'offense_win_rate': wins[0] / num_games,
        'defense_win_rate': wins[1] / num_games,
        'draw_rate': wins[2] / num_games,
    }


def main():
    parser = argparse.ArgumentParser(description='Agent Round Robin Tournament')
    parser.add_argument('--ppo-model', type=str, default='models/ppo_cloud/ppo_nfl-bucketed_final.pt',
                        help='Path to PPO model')
    parser.add_argument('--nfsp-model', type=str, default='models/nfsp_cloud/nfsp_nfl-bucketed_p0_final.pt',
                        help='Path to NFSP model (player 0)')
    parser.add_argument('--dmc-model', type=str, default='experiments/dmc_cloud/dmc_nfl-bucketed',
                        help='Path to DMC model directory')
    parser.add_argument('--num-games', type=int, default=100,
                        help='Number of games per matchup')
    parser.add_argument('--include-random', action='store_true',
                        help='Include random agent in tournament')
    args = parser.parse_args()
    
    # Create environment
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True,
        'use_cached_model': True
    })
    
    print("="*70)
    print("AGENT ROUND ROBIN TOURNAMENT")
    print("="*70)
    print(f"Games per matchup: {args.num_games}")
    print()
    
    # Load agents
    agents = {}
    
    print("Loading agents...")
    
    # PPO
    if os.path.exists(args.ppo_model):
        agents['PPO'] = ('ppo', load_ppo_agent(args.ppo_model, env))
        print(f"  [OK] PPO loaded from {args.ppo_model}")
    else:
        print(f"  [X] PPO model not found: {args.ppo_model}")
    
    # NFSP
    if os.path.exists(args.nfsp_model):
        agents['NFSP'] = ('nfsp', load_nfsp_agent(args.nfsp_model, env))
        print(f"  [OK] NFSP loaded from {args.nfsp_model}")
    else:
        print(f"  [X] NFSP model not found: {args.nfsp_model}")
    
    # DMC
    if os.path.exists(os.path.join(args.dmc_model, 'model.tar')):
        agents['DMC_P0'] = ('dmc', load_dmc_agent(args.dmc_model, env, 0))
        agents['DMC_P1'] = ('dmc', load_dmc_agent(args.dmc_model, env, 1))
        print(f"  [OK] DMC loaded from {args.dmc_model}")
    else:
        print(f"  [X] DMC model not found: {args.dmc_model}")
    
    # Random baseline
    if args.include_random:
        agents['Random'] = ('random', RandomAgent(env.num_actions))
        print("  [OK] Random agent added")
    
    print()
    
    # Define matchups
    # For DMC, use P0 for offense and P1 for defense
    offense_agents = [('PPO', agents.get('PPO')), ('NFSP', agents.get('NFSP'))]
    defense_agents = [('PPO', agents.get('PPO')), ('NFSP', agents.get('NFSP'))]
    
    if 'DMC_P0' in agents:
        offense_agents.append(('DMC', agents.get('DMC_P0')))
        defense_agents.append(('DMC', agents.get('DMC_P1')))
    
    if args.include_random and 'Random' in agents:
        offense_agents.append(('Random', agents.get('Random')))
        defense_agents.append(('Random', agents.get('Random')))
    
    # Filter out None agents
    offense_agents = [(n, a) for n, a in offense_agents if a is not None]
    defense_agents = [(n, a) for n, a in defense_agents if a is not None]
    
    # Run round robin
    results = {}
    
    print("Running tournament...")
    print("-"*70)
    
    for off_name, off_agent in offense_agents:
        for def_name, def_agent in defense_agents:
            matchup = f"{off_name} (O) vs {def_name} (D)"
            print(f"  {matchup}...", end=" ", flush=True)
            
            game_agents = [off_agent[1], def_agent[1]]
            game_types = [off_agent[0], def_agent[0]]
            
            result = run_games(game_agents, game_types, env, args.num_games)
            results[matchup] = result
            
            print(f"EPA: {result['avg_epa_offense']:+.3f} | Win%: {result['offense_win_rate']*100:.0f}%")
    
    # Print results table
    print()
    print("="*70)
    print("RESULTS MATRIX - Offense EPA (Win Rate)")
    print("="*70)
    
    # Create matrix
    off_names = list(set(n for n, _ in offense_agents))
    def_names = list(set(n for n, _ in defense_agents))
    
    # Header
    col_header = "Offense \\ Defense"
    header = f"{col_header:>20}"
    for def_name in def_names:
        header += f" | {def_name:>15}"
    print(header)
    print("-"*len(header))
    
    # Rows
    for off_name in off_names:
        row = f"{off_name:>20}"
        for def_name in def_names:
            matchup = f"{off_name} (O) vs {def_name} (D)"
            if matchup in results:
                r = results[matchup]
                row += f" | {r['avg_epa_offense']:+.2f} ({r['offense_win_rate']*100:.0f}%)"
            else:
                row += " |" + " "*15
        print(row)
    
    # Summary statistics
    print()
    print("="*70)
    print("AGENT RANKINGS")
    print("="*70)
    
    # Calculate average offensive and defensive performance
    off_scores = {n: [] for n in off_names}
    def_scores = {n: [] for n in def_names}
    
    for matchup, result in results.items():
        parts = matchup.split(' vs ')
        off_name = parts[0].replace(' (O)', '')
        def_name = parts[1].replace(' (D)', '')
        
        off_scores[off_name].append(result['avg_epa_offense'])
        def_scores[def_name].append(-result['avg_epa_offense'])  # Defense wants low offense EPA
    
    print("\nOffense Performance (higher = better):")
    for name in sorted(off_scores, key=lambda n: np.mean(off_scores[n]) if off_scores[n] else -999, reverse=True):
        if off_scores[name]:
            print(f"  {name}: {np.mean(off_scores[name]):+.3f} EPA")
    
    print("\nDefense Performance (higher = better at stopping offense):")
    for name in sorted(def_scores, key=lambda n: np.mean(def_scores[n]) if def_scores[n] else -999, reverse=True):
        if def_scores[name]:
            print(f"  {name}: {np.mean(def_scores[name]):+.3f} EPA allowed")


if __name__ == '__main__':
    main()
