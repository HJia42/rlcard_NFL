"""
PPO Self-Play Training for NFL Play-Calling

Trains a PPO agent via naive self-play on the NFL environment.
Both offense and defense are controlled by the same agent.

Usage:
    python examples/run_ppo_nfl.py --game nfl --episodes 100000
    python examples/run_ppo_nfl.py --game nfl-bucketed --episodes 100000
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import rlcard
from rlcard.agents.ppo_agent import PPOAgent


def train_ppo(
    game='nfl-bucketed',
    num_episodes=100000,
    eval_every=1000,
    save_every=5000,
    rollout_size=256,
    hidden_dims=[128, 128],
    lr=3e-4,
    entropy_coef=0.01,
    save_dir='models/ppo_nfl',
):
    """Train PPO agent via self-play.
    
    Args:
        game: 'nfl' or 'nfl-bucketed'
        num_episodes: Total training episodes
        eval_every: Evaluate every N episodes
        save_every: Save checkpoint every N episodes
        rollout_size: Number of steps before PPO update
        hidden_dims: Hidden layer sizes
        lr: Learning rate
        save_dir: Directory to save models
    """
    # Create environment
    print(f"\n{'='*60}")
    print(f"PPO Self-Play Training")
    print(f"{'='*60}")
    print(f"Game: {game}")
    print(f"Episodes: {num_episodes:,}")
    print(f"Rollout size: {rollout_size}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Save directory: {save_dir}")
    print(f"{'='*60}\n")
    
    # Use full drives for bucketed game, single play for full game
    use_single_play = (game == 'nfl')  # Only single play for full game
    
    env = rlcard.make(game, config={'single_play': use_single_play})
    
    # Determine max actions across all phases
    # Phase 0: 7 (formations + special teams)
    # Phase 1: 5 (defense)
    # Phase 2: 2 (play type)
    max_actions = 7  # Maximum action space
    
    # Create PPO agent
    state_shape = env.state_shape[0]
    agent = PPOAgent(
        state_shape=state_shape,
        num_actions=max_actions,
        lr=lr,
        hidden_dims=hidden_dims,
        n_epochs=4,
        batch_size=64,
        clip_epsilon=0.2,
        entropy_coef=entropy_coef,
    )
    
    # Register agent for both players (self-play)
    env.set_agents([agent, agent])
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Training loop
    episode_rewards = []
    step_count = 0
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        # Run one episode
        state, player_id = env.reset()
        
        episode_reward = 0
        
        while not env.is_over():
            # Get action from agent
            action = agent.step(state)
            
            # Step environment
            next_state, next_player_id = env.step(action)
            
            # Get reward (EPA for offense, -EPA for defense)
            if env.is_over():
                payoffs = env.get_payoffs()
                reward = payoffs[player_id]
                
                # Feed final transition
                agent.feed((state, action, reward, next_state, True))
                episode_reward = payoffs[0]  # Track offense EPA
            else:
                # Intermediate step - no immediate reward
                agent.feed((state, action, 0, next_state, False))
            
            state = next_state
            player_id = next_player_id
            step_count += 1
        
        episode_rewards.append(episode_reward)
        
        # Update PPO every rollout_size steps
        if step_count >= rollout_size:
            stats = agent.update()
            step_count = 0
            
            if episode % 100 == 0 and stats:
                print(f"[Update] Policy Loss: {stats['policy_loss']:.4f}, "
                      f"Value Loss: {stats['value_loss']:.4f}, "
                      f"Entropy: {stats['entropy']:.4f}")
        
        # Logging
        if episode % eval_every == 0:
            avg_reward = np.mean(episode_rewards[-eval_every:])
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            
            print(f"\n[Episode {episode:,}] "
                  f"Avg EPA: {avg_reward:.3f}, "
                  f"Speed: {eps_per_sec:.1f} eps/sec")
        
        # Save checkpoint
        if episode % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f'ppo_{game}_{episode}.pt')
            agent.save(checkpoint_path)
    
    # Save final model
    final_path = os.path.join(save_dir, f'ppo_{game}_final.pt')
    agent.save(final_path)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}")
    
    return agent


def evaluate(agent, game='nfl-bucketed', num_episodes=1000):
    """Evaluate trained agent.
    
    Args:
        agent: Trained PPO agent
        game: Game environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Average EPA
    """
    env = rlcard.make(game, config={'single_play': True})
    env.set_agents([agent, agent])
    
    rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        
        while not env.is_over():
            action, _ = agent.eval_step(state)
            state, _ = env.step(action)
        
        rewards.append(env.get_payoffs()[0])
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    print(f"\nEvaluation ({num_episodes} games):")
    print(f"  Average EPA: {avg_reward:.3f} ± {std_reward:.3f}")
    
    return avg_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PPO Self-Play for NFL')
    parser.add_argument('--game', type=str, default='nfl-bucketed',
                        choices=['nfl', 'nfl-bucketed'],
                        help='Game environment')
    parser.add_argument('--episodes', type=int, default=100000,
                        help='Number of training episodes')
    parser.add_argument('--rollout', type=int, default=256,
                        help='Steps before PPO update')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--hidden', type=int, nargs='+', default=[128, 128],
                        help='Hidden layer dimensions')
    parser.add_argument('--save-dir', type=str, default='models/ppo_nfl',
                        help='Save directory')
    parser.add_argument('--entropy-coef', type=float, default=0.01,
                        help='Entropy coefficient for exploration (default: 0.01)')
    parser.add_argument('--eval-only', type=str, default=None,
                        help='Path to model for evaluation only')
    
    args = parser.parse_args()
    
    if args.eval_only:
        # Evaluation mode
        env = rlcard.make(args.game, config={'single_play': True})
        agent = PPOAgent(
            state_shape=env.state_shape[0],
            num_actions=7,
            hidden_dims=args.hidden,
        )
        agent.load(args.eval_only)
        evaluate(agent, args.game)
    else:
        # Training mode
        train_ppo(
            game=args.game,
            num_episodes=args.episodes,
            rollout_size=args.rollout,
            lr=args.lr,
            hidden_dims=args.hidden,
            entropy_coef=args.entropy_coef,
            save_dir=args.save_dir,
        )
