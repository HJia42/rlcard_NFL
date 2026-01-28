
import pandas as pd
import argparse

def analyze_differences(csv_path):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}. Please run analyze_bucketed_policy.py first.")
        return

    # Pivot to compare Standard vs IIG side-by-side
    # We want columns: Agent, Down, Dist, Field, Pass_Prob_Std, Pass_Prob_IIG, Delta
    
    # Filter for interesting columns
    df_subset = df[['Env', 'Agent', 'Down', 'Dist', 'Field', 'Pass_Prob']]
    
    # Separate Standard and IIG
    std_df = df_subset[df_subset['Env'] == 'nfl-bucketed'].rename(columns={'Pass_Prob': 'Pass_Std'})
    iig_df = df_subset[df_subset['Env'] == 'nfl-iig-bucketed'].rename(columns={'Pass_Prob': 'Pass_IIG'})
    
    # Merge
    merged = pd.merge(std_df, iig_df, on=['Agent', 'Down', 'Dist', 'Field'])
    
    # Calculate Delta
    merged['Delta'] = merged['Pass_IIG'] - merged['Pass_Std']
    merged['Abs_Delta'] = merged['Delta'].abs()
    
    # Analyze per Agent
    agents = merged['Agent'].unique()
    
    for agent in agents:
        print(f"\n{'='*80}")
        print(f"Top Strategy Shifts for {agent.upper()}")
        print(f"Positive Delta = More Passing in IIG (Hidden Info)")
        print(f"{'='*80}")
        
        agent_data = merged[merged['Agent'] == agent]
        
        # global stats
        avg_std = agent_data['Pass_Std'].mean()
        avg_iig = agent_data['Pass_IIG'].mean()
        print(f"Average Pass Prob: Standard {avg_std:.3f} -> IIG {avg_iig:.3f} (Delta: {avg_iig-avg_std:.3f})")
        
        print("\n--- Largest Strategy Changes (Top 10) ---")
        top_changes = agent_data.sort_values('Abs_Delta', ascending=False).head(10)
        
        print(f"{'Down':<5} {'Dist':<8} {'Field':<10} | {'Std Pass':<10} {'IIG Pass':<10} | {'Delta':<10}")
        print("-" * 60)
        for _, row in top_changes.iterrows():
            print(f"{row['Down']:<5} {row['Dist']:<8} {row['Field']:<10} | {row['Pass_Std']:<10.3f} {row['Pass_IIG']:<10.3f} | {row['Delta']:<+10.3f}")

if __name__ == "__main__":
    analyze_differences('agent_policy_analysis.csv')
