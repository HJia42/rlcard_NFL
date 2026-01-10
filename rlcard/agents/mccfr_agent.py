"""
Monte Carlo Counterfactual Regret Minimization (MCCFR) Agent

This implements MCCFR with External Sampling, which is more efficient
than vanilla CFR for games with large state/action spaces.

Key difference from vanilla CFR:
- Vanilla CFR: Explores ALL actions for ALL players (full tree traversal)
- MCCFR External: Explores ALL actions for traverser, SAMPLES for opponents

This dramatically reduces the number of nodes visited while maintaining
convergence guarantees.

Reference:
- Lanctot et al., "Monte Carlo Sampling for Regret Minimization in Extensive Games"
"""

import numpy as np
import collections
import os
import pickle

from rlcard.utils.utils import remove_illegal


class MCCFRAgent:
    """Monte Carlo CFR Agent with External Sampling.
    
    External sampling means:
    - For the player we're computing regrets for: explore ALL actions
    - For opponent players: SAMPLE a single action from their policy
    
    This provides significant speedup over vanilla CFR while converging
    to the same Nash equilibrium.
    """

    def __init__(self, env, model_path='./mccfr_model'):
        """Initialize MCCFR Agent.

        Args:
            env (Env): RLCard environment (must support step_back)
            model_path (str): Path for saving/loading model
        """
        self.use_raw = False
        self.env = env
        self.model_path = model_path

        # Policy: state_str -> action probabilities
        self.policy = collections.defaultdict(list)
        self.average_policy = collections.defaultdict(np.array)

        # Regret: state_str -> action regrets
        self.regrets = collections.defaultdict(np.array)

        self.iteration = 0
        
        # Statistics for monitoring
        self.stats = {
            'nodes_visited': 0,
            'nodes_sampled': 0,
            'terminal_nodes': 0,
        }

    def train(self):
        """Do one iteration of MCCFR with external sampling."""
        self.iteration += 1
        
        # Reset stats for this iteration
        self.stats = {'nodes_visited': 0, 'nodes_sampled': 0, 'terminal_nodes': 0}
        
        # Traverse tree for each player
        for player_id in range(self.env.num_players):
            self.env.reset()
            probs = np.ones(self.env.num_players)
            self._traverse_external(probs, player_id)

        # Update policy based on regrets
        self._update_policy()
        
        return self.stats

    def _traverse_external(self, probs, player_id):
        """Traverse game tree using external sampling.
        
        External sampling:
        - Traversing player (player_id): Explore ALL actions
        - Other players: SAMPLE one action from current policy
        
        Args:
            probs: Reach probabilities for each player
            player_id: The player we're computing regrets for
            
        Returns:
            Expected utilities for all players at this node
        """
        self.stats['nodes_visited'] += 1
        
        # Terminal node
        if self.env.is_over():
            self.stats['terminal_nodes'] += 1
            return self.env.get_payoffs()

        current_player = self.env.get_player_id()
        obs, legal_actions = self._get_state(current_player)
        action_probs = self._action_probs(obs, legal_actions, self.policy)

        if current_player == player_id:
            # TRAVERSING PLAYER: Explore ALL actions
            action_utilities = {}
            state_utility = np.zeros(self.env.num_players)

            for action in legal_actions:
                action_prob = action_probs[action]
                new_probs = probs.copy()
                new_probs[current_player] *= action_prob

                # Step, recurse, step back
                self.env.step(action)
                utility = self._traverse_external(new_probs, player_id)
                self.env.step_back()

                state_utility += action_prob * utility
                action_utilities[action] = utility

            # Compute and accumulate regrets
            self._update_regrets(
                obs, legal_actions, action_utilities, 
                state_utility, probs, current_player
            )
            
            return state_utility
        else:
            # OPPONENT: Sample ONE action from policy
            self.stats['nodes_sampled'] += 1
            
            # Sample action according to policy
            action = self._sample_action(action_probs, legal_actions)
            
            # Update reach probability for this player
            new_probs = probs.copy()
            new_probs[current_player] *= action_probs[action]

            # Step, recurse, step back
            self.env.step(action)
            utility = self._traverse_external(new_probs, player_id)
            self.env.step_back()

            return utility

    def _sample_action(self, action_probs, legal_actions):
        """Sample an action from the given probabilities.
        
        Args:
            action_probs: Array of probabilities for each action
            legal_actions: List of legal action indices
            
        Returns:
            Sampled action index
        """
        # Extract probabilities for legal actions only
        legal_probs = np.array([action_probs[a] for a in legal_actions])
        
        # Ensure probabilities sum to 1 (handle numerical issues)
        legal_probs = legal_probs / legal_probs.sum()
        
        # Sample
        action_idx = np.random.choice(len(legal_actions), p=legal_probs)
        return legal_actions[action_idx]

    def _update_regrets(self, obs, legal_actions, action_utilities, 
                        state_utility, probs, current_player):
        """Update regrets for the current state.
        
        Uses Linear CFR weighting (weight by iteration number) for
        faster convergence.
        """
        player_prob = probs[current_player]
        counterfactual_prob = (np.prod(probs[:current_player]) *
                               np.prod(probs[current_player + 1:]))
        player_state_utility = state_utility[current_player]

        # Initialize if first visit
        if obs not in self.regrets:
            self.regrets[obs] = np.zeros(self.env.num_actions)
        if obs not in self.average_policy:
            self.average_policy[obs] = np.zeros(self.env.num_actions)

        # Get current action probs for policy update
        action_probs = self._action_probs(obs, legal_actions, self.policy)
        
        # Update regrets and average policy
        for action in legal_actions:
            action_prob = action_probs[action]
            
            # Regret = counterfactual value of action - counterfactual value of state
            regret = counterfactual_prob * (
                action_utilities[action][current_player] - player_state_utility
            )
            self.regrets[obs][action] += regret
            
            # Linear CFR: weight by iteration for faster convergence
            self.average_policy[obs][action] += (
                self.iteration * player_prob * action_prob
            )

    def _update_policy(self):
        """Update policy using regret matching."""
        for obs in self.regrets:
            self.policy[obs] = self._regret_matching(obs)

    def _regret_matching(self, obs):
        """Apply regret matching to get new policy.
        
        Actions with positive regret get probability proportional to regret.
        If all regrets are non-positive, use uniform distribution.
        """
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

    def _action_probs(self, obs, legal_actions, policy):
        """Get action probabilities for a state.
        
        Args:
            obs: State observation (bytes)
            legal_actions: List of legal action indices
            policy: Policy dictionary to use
            
        Returns:
            Array of action probabilities
        """
        if obs not in policy.keys():
            # Uniform over all actions initially
            action_probs = np.array([1.0/self.env.num_actions 
                                     for _ in range(self.env.num_actions)])
            self.policy[obs] = action_probs
        else:
            action_probs = policy[obs]
        
        # Zero out illegal actions and renormalize
        action_probs = remove_illegal(action_probs, legal_actions)
        return action_probs

    def _get_state(self, player_id):
        """Get state observation for a player.
        
        Args:
            player_id: Player to get state for
            
        Returns:
            Tuple of (obs_key, legal_actions_list)
        """
        state = self.env.get_state(player_id)
        # Use obs_tuple if available (bucketed games), otherwise use obs.tobytes()
        if 'obs_tuple' in state:
            obs_key = state['obs_tuple']
        else:
            obs_key = state['obs'].tobytes()
        return obs_key, list(state['legal_actions'].keys())

    def eval_step(self, state):
        """Predict action based on average policy (for evaluation).
        
        Args:
            state: State dictionary with 'obs' and 'legal_actions'
            
        Returns:
            Tuple of (action, info_dict)
        """
        # Use obs_tuple if available, otherwise obs.tobytes()
        if 'obs_tuple' in state:
            obs_key = state['obs_tuple']
        else:
            obs_key = state['obs'].tobytes()
        legal_actions = list(state['legal_actions'].keys())
        
        probs = self._action_probs(obs_key, legal_actions, self.average_policy)
        action = np.random.choice(len(probs), p=probs)

        info = {}
        info['probs'] = {
            state['raw_legal_actions'][i]: float(probs[list(state['legal_actions'].keys())[i]]) 
            for i in range(len(state['legal_actions']))
        }

        return action, info

    def save(self):
        """Save model to disk."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        with open(os.path.join(self.model_path, 'policy.pkl'), 'wb') as f:
            pickle.dump(self.policy, f)

        with open(os.path.join(self.model_path, 'average_policy.pkl'), 'wb') as f:
            pickle.dump(self.average_policy, f)

        with open(os.path.join(self.model_path, 'regrets.pkl'), 'wb') as f:
            pickle.dump(self.regrets, f)

        with open(os.path.join(self.model_path, 'iteration.pkl'), 'wb') as f:
            pickle.dump(self.iteration, f)

    def load(self):
        """Load model from disk."""
        if not os.path.exists(self.model_path):
            return

        with open(os.path.join(self.model_path, 'policy.pkl'), 'rb') as f:
            self.policy = pickle.load(f)

        with open(os.path.join(self.model_path, 'average_policy.pkl'), 'rb') as f:
            self.average_policy = pickle.load(f)

        with open(os.path.join(self.model_path, 'regrets.pkl'), 'rb') as f:
            self.regrets = pickle.load(f)

        with open(os.path.join(self.model_path, 'iteration.pkl'), 'rb') as f:
            self.iteration = pickle.load(f)
