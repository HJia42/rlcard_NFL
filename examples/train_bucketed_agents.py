"""
Train PPO, NFSP, or DeepCFR on Bucketed NFL Environments.

Usage:
    python examples/train_bucketed_agents.py --agent ppo --env nfl-bucketed
    python examples/train_bucketed_agents.py --agent deep_cfr --env nfl-iig-bucketed
    
Algorithms:
    - PPO: Proximal Policy Optimization (RL)
    - NFSP: Neural Fictitious Self-Play (RL + SL)
    - DeepCFR: Deep Counterfactual Regret Minimization (Regret Matching)
    
Environments:
    - nfl-bucketed: Standard Scrimmage (Offense vs Defense)
    - nfl-iig-bucketed: Imperfect Information Scrimmage
"""

import os
import argparse
import torch
import rlcard
from rlcard.agents.ppo_agent import PPOAgent
from rlcard.agents.nfsp_agent import NFSPAgent
from rlcard.agents.deep_cfr_agent import DeepCFRAgent
from rlcard.utils import (
    tournament,
    Logger,
    plot_curve,
)
import json
import glob

def save_checkpoint(log_dir, episode, agents, agent_type):
    """Save checkpoint with metadata."""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save metadata
    meta = {'episode': episode}
    with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f)
        
    # Save agents
    if agent_type == 'deep_cfr':
        # DeepCFR usually saves to internal path, but we can force it
        agents[0].save_path = checkpoint_dir
        agents[0].save_model(episode) 
    else:
        for idx, agent in enumerate(agents):
            path = os.path.join(checkpoint_dir, f'agent_{idx}.pt')
            agent.save(path)
    
    print(f"\\nSaved checkpoint at episode {episode}")

def load_checkpoint(log_dir, agents, agent_type):
    """Load latest checkpoint if exists."""
    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    meta_path = os.path.join(checkpoint_dir, 'meta.json')
    
    if not os.path.exists(meta_path):
        print("No checkpoint found. Starting from scratch.")
        return 0
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    
    start_episode = meta['episode']
    
    # Load agents
    if agent_type == 'deep_cfr':
        # Check for .pt files
        pt_files = glob.glob(os.path.join(checkpoint_dir, '*.pt'))
        if pt_files:
            agents[0].load(pt_files[0]) 
    else:
        for idx, agent in enumerate(agents):
            path = os.path.join(checkpoint_dir, f'agent_{idx}.pt')
            if os.path.exists(path):
                agent.load(path)
    
    print(f"Resumed from episode {start_episode}")
    return start_episode


def reorganize_trajectories(trajectories, payoffs):
    """Reorganize output of env.run() into (s, a, r, s', done) transitions."""
    transitions_per_player = []
    for p_id, traj in enumerate(trajectories):
        player_transitions = []
        if len(traj) < 3: 
            transitions_per_player.append([])
            continue
            
        # Intermediate steps (reward = 0)
        # Fix: range should exclude the last transition (handled by Final step block)
        # If len=3 (1 step), range(0, 0, 2) -> empty. Correct.
        # If len=5 (2 steps), range(0, 2, 2) -> [0]. Correct.
        for i in range(0, len(traj)-3, 2):
            s = traj[i]
            a = traj[i+1]
            ns = traj[i+2]
            r = 0
            d = False
            player_transitions.append((s, a, r, ns, d))
            
        # Final step (reward = payoff)
        s = traj[-3]
        a = traj[-2]
        ns = traj[-1]
        r = payoffs[p_id]
        d = True
        player_transitions.append((s, a, r, ns, d))
        
        transitions_per_player.append(player_transitions)
    return transitions_per_player


def train(args):
    # Make models directory
    log_dir = args.log_dir
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Environment
    config = {
        'single_play': True,
        'reward_type': args.reward_type,
        'use_distribution_model': True, # Use cached distribution model
        'seed': 42,
        'allow_step_back': args.agent == 'deep_cfr', # DeepCFR requires step_back
        'game_num_players': 2,
    }
    env = rlcard.make(args.env, config=config)
    eval_env = rlcard.make(args.env, config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Initialize Agents
    agents = []
    for i in range(env.num_players):
        if args.agent == 'ppo':
            agent = PPOAgent(
                state_shape=env.state_shape[0], # vector shape
                num_actions=env.num_actions,
                hidden_dims=[128, 128],
                lr=0.0001,
                device=device,
                batch_size=128,
            )
        elif args.agent == 'nfsp':
            agent = NFSPAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                hidden_layers_sizes=[128, 128],
                q_mlp_layers=[128, 128],
                device=device,
            )
        elif args.agent == 'deep_cfr': # Corrected placement for DeepCFR
            # DeepCFR in RLCard assumes standard network sizes often, or takes hidden_layers
            agent = DeepCFRAgent(
                env,
                hidden_layers=[128, 128], 
                batch_size=128,
                train_steps=100, # Steps per iteration
                device=device,
                model_path=log_dir, # Explicitly valid path
            )
        else:
            raise ValueError(f"Unknown agent: {args.agent}")
            
        agents.append(agent)

    env.set_agents(agents)
    eval_env.set_agents(agents)
    
    # Logger
    with Logger(log_dir) as logger:
        
        start_episode = 0
        if args.resume:
            start_episode = load_checkpoint(log_dir, agents, args.agent)
        
        print(f"Start training {args.agent} on {args.env} from episode {start_episode}...")
        
        if args.agent == 'deep_cfr':
            # Deep CFR Loop
            for i in range(start_episode, args.num_episodes): 
                agents[0].train() 
                
                if i % args.evaluate_every == 0:
                    logger.log_performance(i, tournament(eval_env, args.num_eval_games)[0])
                
                if i % args.checkpoint_every == 0 and i > 0:
                    save_checkpoint(log_dir, i, agents, args.agent)
                    
            # Save Final
            agents[0].save_path = log_dir
            agents[0].save_model(args.num_episodes)

        elif args.agent == 'nfsp' or args.agent == 'dqn':
            # NFSP Loop
            for i in range(start_episode, args.num_episodes):
                trajectories, payoffs = env.run(is_training=True)
                transitions = reorganize_trajectories(trajectories, payoffs)
                
                for p_id, trans_list in enumerate(transitions):
                    for ts in trans_list:
                        agents[p_id].feed(ts)

                if i % args.evaluate_every == 0:
                    logger.log_performance(i, tournament(eval_env, args.num_eval_games)[0])

                if i % args.checkpoint_every == 0 and i > 0:
                    save_checkpoint(log_dir, i, agents, args.agent)

            # Save Final
            save_checkpoint(log_dir, args.num_episodes, agents, args.agent)

        elif args.agent == 'ppo':
            # PPO Loop
            step_counter = 0
            for i in range(start_episode, args.num_episodes):
                trajectories, payoffs = env.run(is_training=True)
                transitions = reorganize_trajectories(trajectories, payoffs)
                
                for p_id, trans_list in enumerate(transitions):
                    for ts in trans_list:
                        agents[p_id].feed(ts)
                        step_counter += 1
                
                if step_counter > 256: 
                    for agent in agents:
                        agent.update()
                    step_counter = 0

                if i % args.evaluate_every == 0:
                    logger.log_performance(i, tournament(eval_env, args.num_eval_games)[0])

                if i % args.checkpoint_every == 0 and i > 0:
                    save_checkpoint(log_dir, i, agents, args.agent)
            
            # Save Final
            save_checkpoint(log_dir, args.num_episodes, agents, args.agent)

        # Plot
        csv_path = logger.csv_path
        fig_path = os.path.join(log_dir, 'fig.png')
        plot_curve(csv_path, fig_path, args.agent)
        print(f"Training complete. Saved to {log_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train Bucketed NFL Agents")
    parser.add_argument('--env', type=str, default='nfl-bucketed', choices=['nfl-bucketed', 'nfl-iig-bucketed'])
    parser.add_argument('--agent', type=str, default='ppo', choices=['ppo', 'nfsp', 'deep_cfr'])
    parser.add_argument('--num_episodes', type=int, default=5000)
    parser.add_argument('--num_eval_games', type=int, default=100)
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--checkpoint_every', type=int, default=500, help="Save checkpoint every N episodes")
    parser.add_argument('--resume', action='store_true', help="Resume from last checkpoint if available")
    parser.add_argument(
        '--log_dir', 
        type=str, 
        default='experiments/nfl_experiment',
        help='Directory for saving logs and checkpoints'
    )
    parser.add_argument(
        '--reward_type',
        type=str,
        default='epa',
        choices=['epa', 'yards', 'touchdown', 'score'],
        help='Reward function type (epa, yards, touchdown, score)'
    )
    
    args = parser.parse_args()
    train(args)
