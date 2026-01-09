"""
Train agents on NFL game using RLCard

Usage:
    python examples/nfl_train.py --algorithm dqn --num_episodes 10000
    python examples/nfl_train.py --algorithm dmc
"""

import sys
sys.path.insert(0, '.')

import argparse
import os

import rlcard
from rlcard.agents import DQNAgent, RandomAgent
from rlcard.utils import get_device, reorganize


def train_dqn(args):
    """Train DQN agents."""
    env = rlcard.make('nfl', config={'seed': args.seed})
    eval_env = rlcard.make('nfl', config={'seed': args.seed + 1})
    
    device = get_device()
    
    # Create agents
    agents = []
    for i in range(env.num_players):
        agent = DQNAgent(
            num_actions=env.num_actions,
            state_shape=[11],
            mlp_layers=[128, 128, 128],
            device=device,
            learning_rate=0.0005,
            batch_size=64,
            update_target_estimator_every=1000,
            replay_memory_size=20000,
        )
        agents.append(agent)
    
    env.set_agents(agents)
    eval_env.set_agents(agents)
    
    print(f"Training DQN on NFL game for {args.num_episodes} episodes...")
    print(f"Device: {device}")
    
    for ep in range(1, args.num_episodes + 1):
        # Run one episode
        trajectories, payoffs = env.run(is_training=True)
        
        # Reorganize trajectories into (state, action, reward, next_state, done)
        # The payoff is assigned to the final transition
        trajectories = reorganize(trajectories, payoffs)
        
        # Feed transitions to agents
        for i, agent in enumerate(agents):
            for transition in trajectories[i]:
                agent.feed(transition)
        
        # Evaluate periodically
        if ep % args.eval_every == 0:
            # Run eval games
            eval_payoffs = []
            for _ in range(args.num_eval_games):
                _, p = eval_env.run(is_training=False)
                eval_payoffs.append(p[0])
            
            avg_payoff = sum(eval_payoffs) / len(eval_payoffs)
            print(f"Episode {ep}: Avg Offense EPA = {avg_payoff:.2f}")
        
        # Save periodically
        if ep % args.save_every == 0:
            os.makedirs(args.save_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                save_path = os.path.join(args.save_dir, f'dqn_player_{i}_{ep}.pth')
                agent.save(save_path)
            print(f"Saved models at episode {ep}")
    
    # Final save
    for i, agent in enumerate(agents):
        save_path = os.path.join(args.save_dir, f'dqn_player_{i}_final.pth')
        agent.save(save_path)
    
    print("Training complete!")


def train_dmc(args):
    """Train with Deep Monte Carlo."""
    from rlcard.agents.dmc_agent import DMCTrainer
    
    env = rlcard.make('nfl')
    
    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        xpid='nfl_dmc',
        savedir=args.save_dir,
        save_interval=args.save_interval_minutes,
        num_actor_devices=1,
        num_actors=5,
        training_device="0" if args.cuda else "cpu",
    )
    
    print("Starting DMC training...")
    trainer.start()


def main():
    parser = argparse.ArgumentParser(description='Train agents on NFL')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'dmc'])
    parser.add_argument('--num_episodes', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--num_eval_games', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='models/nfl')
    parser.add_argument('--cuda', type=str, default='')
    parser.add_argument('--save_interval_minutes', type=int, default=30)
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    
    if args.algorithm == 'dqn':
        train_dqn(args)
    elif args.algorithm == 'dmc':
        train_dmc(args)


if __name__ == '__main__':
    main()
