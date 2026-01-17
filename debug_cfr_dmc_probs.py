import torch
import sys
import os
import numpy as np
sys.path.insert(0, '.')
import rlcard
from rlcard.utils.agent_loader import load_agent

def check_probs(agent, name):
    print(f"\nChecking {name}...")
    try:
        env = rlcard.make('nfl')
        state, _ = env.reset()
        
        # Override state for consistency
        env.game.down = 4
        env.game.ydstogo = 10
        env.game.yardline = 25
        state = env.get_state(0)

        # Get action
        _, info = agent.eval_step(state)
        
        if 'probs' in info:
            probs = info['probs']
            print(f"  Raw Probs: {probs}")
            
            # Check sum
            total = sum(probs.values())
            print(f"  Sum: {total}")
            
            # Check keys
            keys = list(probs.keys())
            print(f"  Keys ({len(keys)}): {keys}")
            
            if abs(total - 1.0) < 1e-4:
                print("  [OK] Sums to 1.0")
            else:
                print(f"  [FAIL] Sums to {total}")
                
            if len(keys) == 7:
                 print("  [OK] 7 Actions")
            else:
                 print(f"  [FAIL] {len(keys)} Actions (Expected 7)")
                 
        else:
            print("  [FAIL] 'probs' not in info")
            
    except Exception as e:
        print(f"  [ERROR] {e}")

# Load CFR
cfr_agent, _ = load_agent('cfr', 'models/test_cfr_analysis/cfr_model', verbose=False)
check_probs(cfr_agent, "CFR")

# Load DMC
dmc_agent, _ = load_agent('dmc', 'models/test_dmc_analysis/dmc_nfl', verbose=False)
check_probs(dmc_agent, "DMC")
