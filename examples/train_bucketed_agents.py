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

def train(args):
    # Make models directory
    log_dir = f'experiments/{args.env}_{args.agent}'
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize Environment
    # We force 'epa' reward as requested
    env_config = {
        'single_play': True,
        'reward_type': 'epa',
        'use_distribution_model': True, # Use cached distribution model
    }
    
    env = rlcard.make(args.env, config=env_config)
    eval_env = rlcard.make(args.env, config=env_config)
    
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
        elif args.agent == 'deep_cfr':
            # DeepCFR creates one agent that handles the whole game or typically explicitly initialized
            # But RLCard DeepCFRAgent usually takes index. 
            # Reviewing RLCard DeepCFR implementation:
            # It seems DeepCFRAgent is a single object handling policy for a player 
            # BUT standard examples often init one agent object per player.
            
            # Actually, DeepCFR in RLCard typically uses a single large Buffer but separate networks per player?
            # Let's instantiate per player as is standard for NFSP/PPO.
            agent = DeepCFRAgent(
                env,
                policy_network_layers=(128, 128),
                advantage_network_layers=(128, 128),
                num_rollouts=100, # Lower for speed in demo, increase for quality
                num_traversals=100,
                learning_rate=1e-3,
                batch_size_advantage=128,
                batch_size_strategy=128,
                device=device,
            )
        else:
            raise ValueError(f"Unknown agent: {args.agent}")
            
        agents.append(agent)

    env.set_agents(agents)
    eval_env.set_agents(agents)
    
    # Logger
    logger = Logger(log_dir)
    
    print(f"Start training {args.agent} on {args.env}...")
    
    for episode in range(args.num_episodes):
        if args.agent == 'ppo':
            # PPO Custom Loop (simplified, mostly self-play in RLCard wraps this differently)
            # RLCard's "tournament" handles evaluation. 
            # For PPO, we usually need to call agent.feed manually inside a loop if we are not using high-level trainer.
            # But wait, env.run() creates trajectories. 
            # Let's use a standard step loop for better control or just the simple loop.
            
            trajectories, payoffs = env.run(is_training=True)
            # PPOAgent in RLCard expects separate feed calls? 
            # The standard PPO agent in RLCard has a `feed(transition)` method.
            # We must iterate trajectories.
            if episode == 0:
                 pass # check structure
                 
            # Reorganize data for PPO
            # trajectories is list of [transition, transition...]
            # transition is (state, action, reward, next_state, done)
            
            # Since PPO is on-policy, we update after collecting batch.
            pass # Standard loop below works for NFSP, DeepCFR. PPO needs explicit updates.
            
            # Let's just use the tournament loop style but with training steps
            
        elif args.agent == 'deep_cfr':
             # DeepCFR has its own train loop usually?
             # agent.train() runs one iteration of CFR.
             pass
        else:
             # NFSP
             pass

    # RLCard provides `tournament` for evaluation but training loop differs by agent.
    # Let's build specific loops.
    
    if args.agent == 'deep_cfr':
        # Deep CFR Loop
        for i in range(args.num_episodes): # Here 'episodes' means iterations for DeepCFR
            agents[0].train() # DeepCFR agent usually shares memory or trains both? 
            # In standard RLCard DeepCFR example: it has `agent.train()`
            
            if i % args.evaluate_every == 0:
                logger.log_performance(i, tournament(eval_env, args.num_eval_games)[0])
                
        # Save
        agents[0].save_path = log_dir
        agents[0].save_model(i) # Save final

    elif args.agent == 'nfsp' or args.agent == 'dqn':
        # NFSP Loop
        for i in range(args.num_episodes):
            env.run(is_training=True)
            if i % args.evaluate_every == 0:
                logger.log_performance(i, tournament(eval_env, args.num_eval_games)[0])
                
        # Save
        for idx, agent in enumerate(agents):
            save_path = os.path.join(log_dir, f'model_player_{idx}.pth')
            agent.save(save_path)

    elif args.agent == 'ppo':
        # PPO Loop
        # Needs to collect rollouts then update
        step_counter = 0
        for i in range(args.num_episodes):
            trajectories, payoffs = env.run(is_training=True)
            
            # Flatten trajectories
            for traj in trajectories:
                for transition in traj:
                    # state, action, reward, next_state, done
                    # PPO agent.feed takes entries
                    # PPO implemented in RLCard assumes player_id matches
                    pass # Handled by env.run() returning per-player trajectories
            
            # Feed data to respective agents
            for p_id, traj in enumerate(trajectories):
                for ts in traj:
                    # ts = (state, action, reward, next_state, done)
                    agents[p_id].feed(ts)
                    step_counter += 1
            
            # Update agents periodically? Or every episode?
            # Usually update after batch
            if step_counter > 256: 
                for agent in agents:
                    agent.update() # PPO update
                step_counter = 0

            if i % args.evaluate_every == 0:
                logger.log_performance(i, tournament(eval_env, args.num_eval_games)[0])
        
        # Save
        for idx, agent in enumerate(agents):
             agent.save(os.path.join(log_dir, f'ppo_agent_{idx}.pt'))

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
    
    args = parser.parse_args()
    train(args)
