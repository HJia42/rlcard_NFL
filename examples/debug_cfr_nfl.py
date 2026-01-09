"""
Debug test script for CFR agent with NFL game.

This script provides verbose output showing exactly what the CFR agent
is doing during tree traversal, including:
- Step and step_back operations
- Phase transitions
- Action selection and probabilities
- Regret updates

Usage:
    python examples/debug_cfr_nfl.py                    # Default: 1 iteration, max_depth=6
    python examples/debug_cfr_nfl.py --max_depth 3      # Limit depth for smaller output
    python examples/debug_cfr_nfl.py --quiet            # Only show summary, no tree details
    python examples/debug_cfr_nfl.py --iterations 5     # Run more iterations
"""
import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
import collections
import rlcard
from rlcard.utils.utils import remove_illegal


class DebugCFRAgent:
    """CFR Agent with verbose debugging output."""
    
    def __init__(self, env, verbose=True, max_depth=None):
        self.use_raw = False
        self.env = env
        self.verbose = verbose
        self.max_depth = max_depth  # None = unlimited
        
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)
        self.regrets = collections.defaultdict(np.array)
        self.iteration = 0
        
        # Debug counters
        self.step_count = 0
        self.step_back_count = 0
        self.tree_depth = 0
        self.depth_cutoffs = 0  # Track how many times we hit max depth
        
    def log(self, msg, indent=0):
        """Print debug message with indentation."""
        if self.verbose:
            prefix = "  " * min(indent, 20)  # Cap indentation for readability
            print(f"{prefix}{msg}")
    
    def train(self):
        """Do one iteration of CFR with verbose output."""
        self.iteration += 1
        self.step_count = 0
        self.step_back_count = 0
        
        self.log(f"\n{'='*60}")
        self.log(f"CFR ITERATION {self.iteration}")
        self.log(f"{'='*60}")
        
        for player_id in range(self.env.num_players):
            self.log(f"\n--- Training for Player {player_id} ---")
            self.env.reset()
            self.log(f"Game reset. Current player: {self.env.get_player_id()}")
            self._print_game_state()
            
            probs = np.ones(self.env.num_players)
            self.tree_depth = 0
            self.traverse_tree(probs, player_id)
        
        self.update_policy()
        
        self.log(f"\n--- Iteration {self.iteration} Summary ---")
        self.log(f"Total steps: {self.step_count}")
        self.log(f"Total step_backs: {self.step_back_count}")
        self.log(f"Unique states with regrets: {len(self.regrets)}")
    
    def _print_game_state(self):
        """Print current game state."""
        game = self.env.game
        phase_names = ['FORMATION', 'DEFENSE', 'PLAY_TYPE']
        phase_name = phase_names[game.phase] if game.phase < 3 else 'UNKNOWN'
        
        self.log(f"  Game State: Down {game.down}, {game.ydstogo} to go, at {game.yardline} yard line")
        self.log(f"  Phase: {game.phase} ({phase_name})")
        self.log(f"  Current Player: {game.current_player}")
        self.log(f"  Is Over: {game.is_over_flag}")
        if game.pending_formation:
            self.log(f"  Pending Formation: {game.pending_formation}")
        if game.pending_defense_action:
            self.log(f"  Pending Defense: {game.pending_defense_action}")
        self.log(f"  History stack size: {len(game.history)}")
    
    def traverse_tree(self, probs, player_id):
        """Traverse game tree with verbose logging."""
        self.tree_depth += 1
        depth = self.tree_depth
        
        if self.env.is_over():
            payoffs = self.env.get_payoffs()
            self.log(f"[Depth {depth}] TERMINAL NODE - Payoffs: {payoffs}", depth)
            self.tree_depth -= 1
            return payoffs
        
        current_player = self.env.get_player_id()
        obs, legal_actions = self.get_state(current_player)
        action_probs = self.action_probs(obs, legal_actions, self.policy)
        
        self.log(f"[Depth {depth}] Player {current_player}'s turn, {len(legal_actions)} legal actions", depth)
        
        action_utilities = {}
        state_utility = np.zeros(self.env.num_players)
        
        for action in legal_actions:
            action_prob = action_probs[action]
            new_probs = probs.copy()
            new_probs[current_player] *= action_prob
            
            # Describe the action
            game = self.env.game
            if game.phase == 0:
                action_name = game.formation_actions[action]
            elif game.phase == 1:
                action_name = game.defense_actions[action]
            else:
                action_name = game.play_type_actions[action]
            
            self.log(f"[Depth {depth}] Trying action {action} ({action_name}) with prob {action_prob:.3f}", depth)
            
            # Step
            self.step_count += 1
            self.env.step(action)
            self.log(f"[Depth {depth}]   -> STEP #{self.step_count} taken", depth)
            self._print_game_state()
            
            # Recurse
            utility = self.traverse_tree(new_probs, player_id)
            
            # Step back
            self.step_back_count += 1
            self.env.step_back()
            self.log(f"[Depth {depth}]   <- STEP_BACK #{self.step_back_count}", depth)
            
            state_utility += action_prob * utility
            action_utilities[action] = utility
        
        if current_player != player_id:
            self.tree_depth -= 1
            return state_utility
        
        # Compute regrets for current player
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                               np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]
        
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)
        
        self.log(f"[Depth {depth}] Updating regrets for player {current_player}", depth)
        for action in legal_actions:
            action_prob = action_probs[action]
            regret = counterfactual_prob * (action_utilities[action][current_player] - player_state_utility)
            self.regrets[obs][action] += regret
            self.average_policy[obs][action] += self.iteration * player_prob * action_prob
            
            self.log(f"[Depth {depth}]   Action {action}: regret={regret:.4f}, cumulative={self.regrets[obs][action]:.4f}", depth)
        
        self.tree_depth -= 1
        return state_utility
    
    def update_policy(self):
        """Update policy based on regrets."""
        for obs in self.regrets:
            self.policy[obs] = self.regret_matching(obs)
    
    def regret_matching(self, obs):
        """Apply regret matching."""
        regret = self.regrets[obs]
        positive_regret_sum = sum([r for r in regret if r > 0])
        
        action_probs = np.zeros(self.env.num_actions)
        if positive_regret_sum > 0:
            for action in range(self.env.num_actions):
                action_probs[action] = max(0.0, regret[action] / positive_regret_sum)
        else:
            for action in range(self.env.num_actions):
                action_probs[action] = 1.0 / self.env.num_actions
        return action_probs
    
    def action_probs(self, obs, legal_actions, policy):
        """Get action probabilities."""
        if obs not in policy.keys():
            action_probs = np.array([1.0/self.env.num_actions for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs
    
    def get_state(self, player_id):
        """Get state string for player."""
        state = self.env.get_state(player_id)
        return state['obs'].tobytes(), list(state['legal_actions'].keys())
    
    def eval_step(self, state):
        """Predict action based on average policy."""
        probs = self.action_probs(state['obs'].tobytes(), list(state['legal_actions'].keys()), self.average_policy)
        action = np.random.choice(len(probs), p=probs)
        info = {'probs': dict(enumerate(probs))}
        return action, info


def main():
    print("="*60)
    print("DEBUG CFR NFL TRAINING")
    print("="*60)
    
    # Create environment with step_back enabled
    env = rlcard.make(
        'nfl',
        config={
            'seed': 42,
            'allow_step_back': True,
        }
    )
    
    print(f"\nEnvironment created:")
    print(f"  Game: nfl")
    print(f"  allow_step_back: True")
    print(f"  use_simple_model: {env.game.use_simple_model}")
    print(f"  num_actions: {env.num_actions}")
    print(f"  num_players: {env.num_players}")
    
    # Create debug CFR agent
    agent = DebugCFRAgent(env, verbose=True)
    
    # Run just 1-2 iterations to see detailed output
    num_iterations = 2
    print(f"\nRunning {num_iterations} CFR iterations with verbose output...")
    
    for i in range(num_iterations):
        agent.train()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Total iterations: {agent.iteration}")
    print(f"Unique states discovered: {len(agent.regrets)}")
    
    # Show learned policy for a few states
    print("\nSample learned policies:")
    for i, (obs, probs) in enumerate(agent.policy.items()):
        if i >= 5:
            print("  ...")
            break
        # Only show non-zero probabilities
        non_zero = [(j, p) for j, p in enumerate(probs) if p > 0.01]
        print(f"  State {i}: {non_zero}")


if __name__ == '__main__':
    main()
