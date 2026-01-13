"""Analyze NFSP agents on 4th down scenarios.

Usage:
    python examples/analyze_nfsp.py models/nfsp_nfl/nfsp_nfl-bucketed_p0_final.pt
"""
import sys
sys.path.insert(0, '.')
import argparse
import torch
import rlcard
from rlcard.agents import NFSPAgent
from rlcard.games.nfl.game_bucketed import INITIAL_ACTIONS


def analyze_nfsp(checkpoint_path, use_cached=True):
    """Load and analyze NFSP agent on key scenarios."""
    
    # Create environment
    env = rlcard.make('nfl-bucketed', config={
        'single_play': True,
        'use_cached_model': use_cached,
    })
    
    # Load checkpoint and create agent
    print(f"Loading NFSP agent from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    agent = NFSPAgent.from_checkpoint(checkpoint)
    
    print('=' * 60)
    print('NFSP Agent - 4th Down Analysis')
    print('=' * 60)
    
    # Test scenarios: (yardline, ydstogo, down, label, expected)
    tests = [
        (25, 10, 1, '1st & 10 own 25', 'GO'),
        (75, 10, 1, '1st & 10 opp 25', 'GO'),
        (25, 10, 4, '4th & 10 own 25', 'PUNT'),
        (75, 10, 4, '4th & 10 opp 25', 'FG'),
        (35, 1, 4, '4th & 1 own 35', 'GO'),
        (95, 3, 4, '4th & Goal at 5', 'GO'),
    ]
    
    correct = 0
    for yardline, ydstogo, down, label, expected in tests:
        env.reset()
        env.game.down = down
        env.game.ydstogo = ydstogo
        env.game.yardline = yardline
        env.game.phase = 0
        state = env.get_state(0)
        
        action, info = agent.eval_step(state)
        probs = info.get('probs', {})
        
        # Get top action
        top_action = max(probs, key=probs.get)
        top_prob = probs[top_action] * 100
        
        # Check if correct
        if expected == 'GO':
            is_correct = top_action not in ['PUNT', 'FG']
        else:
            is_correct = top_action == expected
        
        status = '[OK]' if is_correct else '[X]'
        if is_correct:
            correct += 1
        
        print(f'\n{label} (want {expected}):')
        for action_name, prob in sorted(probs.items(), key=lambda x: -x[1]):
            if prob > 0.01:
                marker = '<--' if action_name == expected else ''
                print(f'  {action_name}: {prob*100:.1f}% {marker}')
        print(f'  Result: {top_action} {status}')
    
    print('\n' + '=' * 60)
    print(f'Score: {correct}/{len(tests)} correct')
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze NFSP agent')
    parser.add_argument('checkpoint', type=str, nargs='?',
                        default='models/nfsp_nfl/nfsp_nfl-bucketed_p0_final.pt',
                        help='Path to NFSP checkpoint file')
    parser.add_argument('--no-cached', action='store_true',
                        help='Disable cached outcome model')
    
    args = parser.parse_args()
    analyze_nfsp(args.checkpoint, use_cached=not args.no_cached)
