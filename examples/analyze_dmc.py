"""
Analyze Trained DMC Model

Load a trained DMC model and visualize:
1. Training history (loss over time)
2. Decision policies for different game states
3. Action value heatmaps

Usage:
    python examples/analyze_dmc.py --model_path experiments/dmc_result/nfl_dmc
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import rlcard
from rlcard.agents.dmc_agent.model import DMCAgent, DMCNet


def load_training_history(model_path):
    """Load training logs from CSV."""
    logs_path = os.path.join(model_path, 'logs.csv')
    if os.path.exists(logs_path):
        # CSV has a comment header starting with #, skip it
        df = pd.read_csv(logs_path, comment='#', skip_blank_lines=True)
        # Remove any rows with NaN in critical columns
        if 'frames' in df.columns:
            df = df.dropna(subset=['frames'])
            df['frames'] = pd.to_numeric(df['frames'], errors='coerce')
        return df
    return None


def load_dmc_model(model_path, env):
    """Load trained DMC model."""
    model_file = os.path.join(model_path, 'model.tar')
    if not os.path.exists(model_file):
        print(f"No model found at {model_file}")
        return None
    
    checkpoint = torch.load(model_file, map_location='cpu')
    
    # model_state_dict is a list with one dict per player
    model_state_dicts = checkpoint['model_state_dict']
    
    # Create agents from checkpoint
    agents = []
    for player_id in range(env.num_players):
        state_shape = env.state_shape[player_id]
        # Action shape is one-hot encoding of num_actions
        action_shape = (env.num_actions,)
        
        agent = DMCAgent(
            state_shape=state_shape,
            action_shape=action_shape,
            mlp_layers=[512, 512, 512, 512, 512],
            device='cpu'
        )
        # Load weights from list
        agent.net.load_state_dict(model_state_dicts[player_id])
        agent.net.eval()
        agents.append(agent)
    
    print(f"Loaded model with {int(checkpoint.get('frames', 0)):,} frames")
    return agents


def analyze_decisions(agent, env, player_id=0):
    """Analyze agent decisions for various game states."""
    # Must match FORMATIONS in game.py
    formations = ["SHOTGUN", "SINGLEBACK", "UNDER CENTER", "I_FORM", "EMPTY"]
    box_counts = [4, 5, 6, 7, 8]
    
    print("\n" + "=" * 70)
    print(f"PLAYER {player_id} DECISION ANALYSIS")
    print("=" * 70)
    
    # Different game situations
    situations = [
        (1, 10, 25, "1st & 10 at own 25"),
        (2, 5, 50, "2nd & 5 at midfield"),
        (3, 2, 75, "3rd & 2 at opp 25"),
        (3, 10, 50, "3rd & 10 at midfield"),
        (4, 1, 60, "4th & 1 at opp 40"),
        (1, 10, 90, "1st & Goal at 10"),
    ]
    
    if player_id == 0:
        # Offense decisions
        print("\n--- FORMATION SELECTION (Phase 0) ---")
        print(f"{'Situation':<25} | Best Formation | Confidence")
        print("-" * 60)
        
        for down, ydstogo, yardline, desc in situations:
            state = create_offense_state_phase0(down, ydstogo, yardline)
            action, probs, values = get_action_and_values(agent, state, env.num_actions)
            best_formation = formations[action] if action < len(formations) else "?"
            print(f"{desc:<25} | {best_formation:<14} | {probs[action]:.1%}")
        
        print("\n--- PLAY TYPE SELECTION (Phase 2) ---")
        print("Given: 1st & 10 at own 25, SHOTGUN formation")
        print(f"{'Box Count':<12} | Rush Value | Pass Value | Best Choice")
        print("-" * 60)
        
        for box in box_counts:
            state = create_offense_state_phase2(1, 10, 25, 0, box)  # SHOTGUN=0
            action, probs, values = get_action_and_values(agent, state, env.num_actions)
            rush_val = values[0] if len(values) > 0 else 0
            pass_val = values[1] if len(values) > 1 else 0
            choice = "PASS" if action == 1 else "RUSH"
            print(f"{box:<12} | {rush_val:+.3f}     | {pass_val:+.3f}     | {choice}")
    
    else:
        # Defense decisions
        # Defense has 5 actions (0-4) mapping to box counts 4-8
        # But network expects 7-dim action one-hot, so evaluate all 7 then pick from first 5
        num_defense_actions = 5
        
        print("\n--- DEFENSE BOX COUNT SELECTION ---")
        print("Given different offensive formations at 1st & 10 midfield:")
        print(f"{'Formation':<12} | Best Box | Confidence")
        print("-" * 50)
        
        for i, formation in enumerate(formations):
            state = create_defense_state(1, 10, 50, i)
            # Evaluate all 7 actions (network size), but only consider first 5 for defense
            _, all_probs, all_values = get_action_and_values(agent, state, env.num_actions)
            # Only consider defense-valid actions (0-4)
            defense_values = all_values[:num_defense_actions]
            action = np.argmax(defense_values)
            # Recalculate probs for defense actions only
            exp_vals = np.exp(defense_values - defense_values.max())
            probs = exp_vals / exp_vals.sum()
            box = action + 4  # Actions 0-4 map to box counts 4-8
            print(f"{formation:<12} | {box:<8} | {probs[action]:.1%}")


def create_offense_state_phase0(down, ydstogo, yardline):
    """Create state array for offense formation phase."""
    obs = np.zeros(12, dtype=np.float32)
    obs[0] = down / 4.0
    obs[1] = min(ydstogo, 30) / 30.0
    obs[2] = yardline / 100.0
    obs[11] = 0.0  # Phase = 0 (formation)
    return obs


def create_offense_state_phase2(down, ydstogo, yardline, formation_idx, box_count):
    """Create state array for offense play type phase."""
    obs = np.zeros(12, dtype=np.float32)
    obs[0] = down / 4.0
    obs[1] = min(ydstogo, 30) / 30.0
    obs[2] = yardline / 100.0
    # Formation one-hot (indices 3-9)
    obs[3 + formation_idx] = 1.0
    # Box count normalized
    obs[10] = (box_count - 4) / 4.0
    obs[11] = 1.0  # Phase = 2 (play type)
    return obs


def create_defense_state(down, ydstogo, yardline, formation_idx):
    """Create state array for defense."""
    obs = np.zeros(12, dtype=np.float32)
    obs[0] = down / 4.0
    obs[1] = min(ydstogo, 30) / 30.0
    obs[2] = yardline / 100.0
    # Formation one-hot
    obs[3 + formation_idx] = 1.0
    obs[11] = 0.5  # Phase = 1 (defense)
    return obs


def get_action_and_values(agent, state, num_actions):
    """Get action, probabilities, and Q-values from agent."""
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    
    with torch.no_grad():
        # Get Q-values for all actions
        values = []
        for action_idx in range(num_actions):
            action_one_hot = np.zeros(num_actions, dtype=np.float32)
            action_one_hot[action_idx] = 1.0
            action_tensor = torch.FloatTensor(action_one_hot).unsqueeze(0)
            
            q_value = agent.net(state_tensor, action_tensor).item()
            values.append(q_value)
    
    values = np.array(values)
    
    # Convert to probabilities using softmax
    exp_values = np.exp(values - values.max())
    probs = exp_values / exp_values.sum()
    
    # Best action
    action = np.argmax(values)
    
    return action, probs, values


def plot_training_history(df, save_path=None):
    """Plot training metrics over time."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Frames over time
        if 'frames' in df.columns:
            axes[0, 0].plot(df['frames'])
            axes[0, 0].set_title('Frames Processed')
            axes[0, 0].set_xlabel('Log Entry')
            axes[0, 0].set_ylabel('Frames')
        
        # Loss
        if 'loss_0' in df.columns:
            axes[0, 1].plot(df['loss_0'], label='Player 0')
            if 'loss_1' in df.columns:
                axes[0, 1].plot(df['loss_1'], label='Player 1')
            axes[0, 1].set_title('Training Loss')
            axes[0, 1].legend()
        
        # Returns
        if 'mean_episode_return_0' in df.columns:
            axes[1, 0].plot(df['mean_episode_return_0'], label='Offense')
            if 'mean_episode_return_1' in df.columns:
                axes[1, 0].plot(df['mean_episode_return_1'], label='Defense')
            axes[1, 0].set_title('Mean Episode Return (EPA)')
            axes[1, 0].axhline(y=0, color='gray', linestyle='--')
            axes[1, 0].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()
            
    except ImportError:
        print("matplotlib not available, skipping plots")


def main():
    parser = argparse.ArgumentParser(description='Analyze DMC Model')
    parser.add_argument('--model_path', type=str, 
                        default='experiments/dmc_result/nfl_dmc',
                        help='Path to DMC experiment folder')
    parser.add_argument('--plot', action='store_true',
                        help='Show training history plots')
    args = parser.parse_args()
    
    print("=" * 70)
    print("DMC MODEL ANALYSIS")
    print("=" * 70)
    print(f"Model path: {args.model_path}")
    
    # Load training history
    print("\n--- Training History ---")
    df = load_training_history(args.model_path)
    if df is not None:
        print(f"Log entries: {len(df)}")
        if 'frames' in df.columns:
            print(f"Total frames: {int(df['frames'].max()):,}")
        if 'loss_0' in df.columns:
            print(f"Final loss (P0): {df['loss_0'].iloc[-1]:.4f}")
        if 'loss_1' in df.columns:
            print(f"Final loss (P1): {df['loss_1'].iloc[-1]:.4f}")
        if 'mean_episode_return_0' in df.columns:
            print(f"Final return (Offense): {df['mean_episode_return_0'].iloc[-1]:.3f}")
        if 'mean_episode_return_1' in df.columns:
            print(f"Final return (Defense): {df['mean_episode_return_1'].iloc[-1]:.3f}")
        
        if args.plot:
            plot_training_history(df)
    else:
        print("No training logs found")
    
    # Load model and analyze decisions
    print("\n--- Loading Model ---")
    env = rlcard.make('nfl-bucketed', config={'single_play': True, 'use_cached_model': True})
    agents = load_dmc_model(args.model_path, env)
    
    if agents:
        print(f"Loaded {len(agents)} agents")
        for player_id, agent in enumerate(agents):
            analyze_decisions(agent, env, player_id)
    else:
        print("Could not load model")


if __name__ == '__main__':
    main()
