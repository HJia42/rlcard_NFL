"""
Agent Comparison and Analysis Tools

Compare trained agents across:
1. Head-to-head performance
2. Exploitability (vs best response)
3. Decision-making visualization

Usage:
    python examples/compare_agents.py --game nfl-bucketed
    python examples/compare_agents.py --game nfl-bucketed --situation
"""

import os
import sys
import argparse
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament


# ========== Agent Loading ==========

def load_cfr_agent(game, model_path='experiments/nfl_cfr/cfr_model'):
    """Load CFR agent."""
    from rlcard.agents import CFRAgent
    env = rlcard.make(game, config={'allow_step_back': True, 'single_play': True})
    agent = CFRAgent(env, model_path)
    try:
        agent.load()
        print(f"  Loaded CFR from {model_path}")
        return agent
    except:
        print(f"  CFR model not found at {model_path}")
        return None


def load_mccfr_agent(game, model_path='models/mccfr'):
    """Load MCCFR agent."""
    from rlcard.agents.mccfr_agent import MCCFRAgent
    env = rlcard.make(game, config={'allow_step_back': True, 'single_play': True})
    agent = MCCFRAgent(env, model_path)
    try:
        agent.load()
        print(f"  Loaded MCCFR from {model_path}")
        return agent
    except:
        print(f"  MCCFR model not found at {model_path}")
        return None


def load_nfsp_agent(game, model_path='models/nfsp_nfl'):
    """Load NFSP agent (player 0)."""
    from rlcard.agents import NFSPAgent
    env = rlcard.make(game, config={'single_play': True})
    agent = NFSPAgent(
        num_actions=env.num_actions,
        state_shape=[11],
        hidden_layers_sizes=[128, 128],
        q_mlp_layers=[128, 128],
    )
    try:
        # Try different naming conventions
        for pattern in [f'nfsp_{game}_p0_final.pt', 'nfsp_player_0_final.pt']:
            path = os.path.join(model_path, pattern)
            if os.path.exists(path):
                agent.load_checkpoint(model_path, filename=pattern)
                print(f"  Loaded NFSP from {path}")
                return agent
        print(f"  NFSP model not found in {model_path}")
        return None
    except Exception as e:
        print(f"  NFSP load error: {e}")
        return None


def load_ppo_agent(game, model_path='models/ppo_nfl'):
    """Load PPO agent."""
    from rlcard.agents.ppo_agent import PPOAgent
    env = rlcard.make(game, config={'single_play': True})
    agent = PPOAgent(
        state_shape=env.state_shape[0],
        num_actions=7,
        hidden_dims=[128, 128],
    )
    try:
        for pattern in [f'ppo_{game}_final.pt', 'ppo_nfl-bucketed_final.pt']:
            path = os.path.join(model_path, pattern)
            if os.path.exists(path):
                agent.load(path)
                print(f"  Loaded PPO from {path}")
                return agent
        print(f"  PPO model not found in {model_path}")
        return None
    except Exception as e:
        print(f"  PPO load error: {e}")
        return None


def load_all_agents(game):
    """Load all available trained agents."""
    print("\nLoading trained agents...")
    
    agents = {}
    
    # CFR
    cfr = load_cfr_agent(game)
    if cfr:
        agents['CFR'] = cfr
    
    # MCCFR
    mccfr = load_mccfr_agent(game)
    if mccfr:
        agents['MCCFR'] = mccfr
    
    # NFSP
    nfsp = load_nfsp_agent(game)
    if nfsp:
        agents['NFSP'] = nfsp
    
    # PPO
    ppo = load_ppo_agent(game)
    if ppo:
        agents['PPO'] = ppo
    
    # Random baseline
    env = rlcard.make(game, config={'single_play': True})
    agents['Random'] = RandomAgent(num_actions=env.num_actions)
    
    print(f"\nLoaded {len(agents)} agents: {list(agents.keys())}")
    return agents


# ========== Head-to-Head Comparison ==========

def head_to_head(agent1, agent2, game, num_games=500):
    """Play agent1 vs agent2 and return EPA difference."""
    env = rlcard.make(game, config={'single_play': True})
    
    # Agent1 as offense
    env.set_agents([agent1, agent2])
    result1 = tournament(env, num_games)
    off_epa = result1[0]
    
    # Agent1 as defense
    env.set_agents([agent2, agent1])
    result2 = tournament(env, num_games)
    def_epa = result2[1]
    
    return {
        'offense_epa': off_epa,
        'defense_epa': def_epa,
        'total_epa': off_epa + def_epa,
    }


def round_robin_tournament(agents, game, num_games=200):
    """Run round-robin tournament between all agents."""
    print("\n" + "=" * 60)
    print("ROUND ROBIN TOURNAMENT")
    print("=" * 60)
    
    names = list(agents.keys())
    results = defaultdict(lambda: {'wins': 0, 'losses': 0, 'total_epa': 0})
    
    matchups = []
    
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            print(f"\n{name1} vs {name2}...")
            
            result = head_to_head(agents[name1], agents[name2], game, num_games)
            
            # Agent1 is winner if total EPA > 0
            if result['total_epa'] > 0:
                results[name1]['wins'] += 1
                results[name2]['losses'] += 1
            else:
                results[name2]['wins'] += 1
                results[name1]['losses'] += 1
            
            results[name1]['total_epa'] += result['total_epa']
            results[name2]['total_epa'] -= result['total_epa']
            
            matchups.append({
                'agent1': name1,
                'agent2': name2,
                'agent1_off_epa': result['offense_epa'],
                'agent1_def_epa': result['defense_epa'],
            })
            
            print(f"  {name1} Off: {result['offense_epa']:.3f} | Def: {result['defense_epa']:.3f}")
    
    # Print standings
    print("\n" + "-" * 40)
    print("STANDINGS")
    print("-" * 40)
    print(f"{'Agent':<12} {'W':>4} {'L':>4} {'EPA':>8}")
    print("-" * 40)
    
    sorted_agents = sorted(results.items(), key=lambda x: (-x[1]['wins'], -x[1]['total_epa']))
    for name, stats in sorted_agents:
        print(f"{name:<12} {stats['wins']:>4} {stats['losses']:>4} {stats['total_epa']:>8.3f}")
    
    return matchups, dict(results)


# ========== Exploitability ==========

def calculate_exploitability(agent, game, num_samples=1000):
    """
    Estimate exploitability by sampling states and computing regret.
    
    True exploitability requires computing exact best response,
    which is expensive. This is an approximation.
    """
    env = rlcard.make(game, config={'single_play': True})
    
    # Compare to random baseline
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    env.set_agents([agent, random_agent])
    off_vs_random = tournament(env, num_samples)[0]
    
    env.set_agents([random_agent, agent])
    def_vs_random = tournament(env, num_samples)[1]
    
    # Exploitability estimate: how much better than random
    # Lower is more exploitable (random exploits it)
    return {
        'vs_random_total': off_vs_random + def_vs_random,
        'offense_margin': off_vs_random,
        'defense_margin': def_vs_random,
    }


def compare_exploitability(agents, game):
    """Compare exploitability across agents."""
    print("\n" + "=" * 60)
    print("EXPLOITABILITY ANALYSIS")
    print("=" * 60)
    print("(Higher = less exploitable, better generalization)")
    print()
    
    results = {}
    
    for name, agent in agents.items():
        if name == 'Random':
            continue
        
        print(f"Analyzing {name}...")
        exp = calculate_exploitability(agent, game)
        results[name] = exp
        print(f"  vs Random: {exp['vs_random_total']:.3f} "
              f"(Off: {exp['offense_margin']:.3f}, Def: {exp['defense_margin']:.3f})")
    
    return results


# ========== Decision Visualization ==========

def get_action_probs(agent, state, agent_name=None):
    """Get action probabilities from agent for given state.
    
    All agents now return a standardized format:
    {'probs': {'action_name': probability, ...}}
    """
    try:
        action, info = agent.eval_step(state)
        
        # Standard format: info['probs'] with str keys
        if isinstance(info, dict) and 'probs' in info:
            return info['probs']
        
        # Fallback: info might be probs dict directly (legacy)
        if isinstance(info, dict) and all(isinstance(v, (int, float)) for v in info.values()):
            return info
        
        if agent_name:
            print(f"{agent_name}: No probs in info (type={type(info)})")
        return None
    except Exception as e:
        if agent_name:
            print(f"{agent_name}: Error - {e}")
        return None


def analyze_situation(agents, game, down, ydstogo, yardline, phase=0):
    """Show how each agent would decide in a specific situation."""
    print(f"\n{'='*60}")
    print(f"SITUATION: {down} & {ydstogo} at own {yardline}")
    print(f"{'='*60}")
    
    env = rlcard.make(game, config={'single_play': True})
    
    # Set up the situation
    env.reset()
    env.game.down = down
    env.game.ydstogo = ydstogo
    env.game.yardline = yardline
    env.game.phase = phase
    
    # Get state from game
    state = env.game.get_state(0)
    
    # For bucketed game, also add obs_tuple for CFR/MCCFR
    if game == 'nfl-bucketed':
        from rlcard.games.nfl.game_bucketed import INITIAL_ACTIONS, get_field_position_bucket, get_distance_bucket
        # Build the obs_tuple that CFR/MCCFR use as keys
        field_bucket = get_field_position_bucket(yardline)
        dist_bucket = get_distance_bucket(ydstogo)
        obs_tuple = (field_bucket, down, dist_bucket, phase)
        state['obs_tuple'] = obs_tuple
        print(f"Bucketed State: field={field_bucket}, down={down}, dist={dist_bucket}, phase={phase}")
    else:
        from rlcard.games.nfl.game import INITIAL_ACTIONS
    
    # Ensure legal_actions is a dict (CFR/MCCFR requirement)
    num_actions = len(INITIAL_ACTIONS)
    if not isinstance(state.get('legal_actions'), dict):
        state['legal_actions'] = {i: None for i in range(num_actions)}
        state['raw_legal_actions'] = list(range(num_actions))
    
    print(f"\nPhase {phase} Actions: {INITIAL_ACTIONS}")
    print()
    
    for name, agent in agents.items():
        if name == 'Random':
            continue
        
        try:
            probs = get_action_probs(agent, state, agent_name=name)
            if probs:
                print(f"{name}:")
                if isinstance(probs, dict):
                    # Sort by probability descending
                    sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
                    for action_key, prob in sorted_probs:
                        if prob > 0.01:
                            # CFR/MCCFR use action names as keys
                            if isinstance(action_key, str):
                                print(f"  {action_key}: {prob:.1%}")
                            # PPO/others use indices
                            elif isinstance(action_key, int) and action_key < len(INITIAL_ACTIONS):
                                print(f"  {INITIAL_ACTIONS[action_key]}: {prob:.1%}")
                            else:
                                print(f"  Action {action_key}: {prob:.1%}")
                else:
                    for i, prob in enumerate(probs):
                        if prob > 0.01:
                            action_name = INITIAL_ACTIONS[i] if i < len(INITIAL_ACTIONS) else f"Action {i}"
                            print(f"  {action_name}: {prob:.1%}")
                print()
        except Exception as e:
            print(f"{name}: Error - {e}")


def analyze_key_situations(agents, game):
    """Analyze agent decisions in key NFL situations."""
    print("\n" + "=" * 60)
    print("KEY SITUATION ANALYSIS")
    print("=" * 60)
    
    situations = [
        # (down, ydstogo, yardline, description)
        (4, 1, 35, "4th & 1 at own 35 - Go for it?"),
        (4, 10, 25, "4th & 10 at own 25 - Punt situation"),
        (4, 3, 65, "4th & 3 at opp 35 - FG range?"),
        (4, 1, 95, "4th & Goal at 5 - TD or FG?"),
        (1, 10, 50, "1st & 10 at midfield"),
        (3, 15, 30, "3rd & 15 at own 30"),
    ]
    
    for down, ydstogo, yardline, desc in situations:
        print(f"\n>>> {desc}")
        analyze_situation(agents, game, down, ydstogo, yardline)


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser(description='Compare NFL Agents')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed', 'nfl-iig', 'nfl-iig-bucketed'])
    parser.add_argument('--num-games', type=int, default=200,
                        help='Games per matchup')
    parser.add_argument('--situation', action='store_true',
                        help='Show situation analysis')
    parser.add_argument('--exploitability', action='store_true',
                        help='Calculate exploitability')
    parser.add_argument('--tournament', action='store_true',
                        help='Run round-robin tournament')
    parser.add_argument('--all', action='store_true',
                        help='Run all analyses')
    
    args = parser.parse_args()
    
    # Load all agents
    agents = load_all_agents(args.game)
    
    if len(agents) <= 1:
        print("\nNot enough trained agents found. Train some agents first!")
        return
    
    # Run requested analyses
    if args.all or args.tournament:
        round_robin_tournament(agents, args.game, args.num_games)
    
    if args.all or args.exploitability:
        compare_exploitability(agents, args.game)
    
    if args.all or args.situation:
        analyze_key_situations(agents, args.game)
    
    if not any([args.all, args.tournament, args.exploitability, args.situation]):
        # Default: show everything
        round_robin_tournament(agents, args.game, args.num_games)
        compare_exploitability(agents, args.game)
        analyze_key_situations(agents, args.game)


if __name__ == '__main__':
    main()
