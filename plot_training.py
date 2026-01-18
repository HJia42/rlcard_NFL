
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_logs(model_dirs, output_file='training_curves.png'):
    plt.figure(figsize=(12, 6))
    
    for label, dir_path in model_dirs.items():
        log_path = os.path.join(dir_path, 'eval_log.csv')
        if not os.path.exists(log_path):
            print(f"Warning: No log found at {log_path}")
            continue
            
        try:
            df = pd.read_csv(log_path)
            # Standard RLCard eval_log has columns like 'timestep', 'reward'
            # Or sometimes just headerless? Let's check typical RLCard Logger.
            # It usually has 'timestep', 'reward' in header.
            
            # Identify columns
            x_col = 'timestep' if 'timestep' in df.columns else df.columns[0]
            y_col = 'reward' if 'reward' in df.columns else df.columns[1]
            
            # Smooth data
            df['smoothed'] = df[y_col].rolling(window=5).mean()
            
            plt.plot(df[x_col], df['smoothed'], label=f"{label} (smoothed)")
            plt.plot(df[x_col], df[y_col], alpha=0.3, label=f"{label} (raw)") # Faint raw line
            
        except Exception as e:
            print(f"Error reading {log_path}: {e}")

    plt.xlabel('Timesteps / Episodes')
    plt.ylabel('Average Reward')
    plt.title('Agent Training Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_file)
    print(f"Saved plot to {output_file}")

if __name__ == "__main__":
    # Define your agent paths here
    agent_paths = {
        'PPO': 'models/ppo_nfl',
        'NFSP': 'models/nfsp_nfl',
        # 'Deep CFR': 'models/deep_cfr_nfl' 
    }
    
    plot_logs(agent_paths)
