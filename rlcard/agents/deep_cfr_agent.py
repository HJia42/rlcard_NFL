"""
Deep Counterfactual Regret Minimization (Deep CFR) Agent

Deep CFR uses neural networks to approximate regrets instead of tabular storage,
enabling generalization across similar game states and handling large state spaces.

Key Components:
- AdvantageNetwork: Predicts per-action regrets/advantages given state
- PolicyNetwork: Predicts average strategy from state
- ReservoirBuffer: Stores training samples using reservoir sampling

Reference:
- Brown et al., "Deep Counterfactual Regret Minimization" (ICML 2019)
"""

import numpy as np
import collections
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlcard.utils.utils import remove_illegal


# Named tuple for advantage samples
AdvantageSample = collections.namedtuple('AdvantageSample', 
    ['state', 'advantages', 'iteration'])

# Named tuple for policy samples  
PolicySample = collections.namedtuple('PolicySample',
    ['state', 'policy'])


class AdvantageNetwork(nn.Module):
    """Neural network that predicts per-action advantages (regrets).
    
    Input: state features
    Output: advantage value for each action
    """
    
    def __init__(self, state_shape, num_actions, hidden_layers=None):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        
        # Build MLP
        input_dim = int(np.prod(state_shape))
        layer_dims = [input_dim] + hidden_layers + [num_actions]
        
        layers = [nn.Flatten()]
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation on final layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        """Predict advantages for all actions."""
        return self.network(state)


class PolicyNetwork(nn.Module):
    """Neural network that predicts average strategy.
    
    Input: state features
    Output: probability distribution over actions
    """
    
    def __init__(self, state_shape, num_actions, hidden_layers=None):
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        
        self.state_shape = state_shape
        self.num_actions = num_actions
        
        # Build MLP
        input_dim = int(np.prod(state_shape))
        layer_dims = [input_dim] + hidden_layers + [num_actions]
        
        layers = [nn.Flatten()]
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation on final layer
                layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, state):
        """Predict log probabilities for all actions."""
        logits = self.network(state)
        return F.log_softmax(logits, dim=-1)


class ReservoirBuffer:
    """Reservoir sampling buffer for storing training samples."""
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = []
        self.add_count = 0
    
    def add(self, sample):
        """Add sample using reservoir sampling."""
        if len(self.data) < self.capacity:
            self.data.append(sample)
        else:
            idx = random.randint(0, self.add_count)
            if idx < self.capacity:
                self.data[idx] = sample
        self.add_count += 1
    
    def sample(self, n):
        """Sample n elements uniformly."""
        if len(self.data) < n:
            return self.data[:]
        return random.sample(self.data, n)
    
    def clear(self):
        """Clear the buffer."""
        self.data = []
        self.add_count = 0
    
    def __len__(self):
        return len(self.data)


class DeepCFRAgent:
    """Deep CFR Agent with Neural Network Regret Approximation.
    
    Uses neural networks instead of tabular regrets, enabling:
    - Generalization across similar states
    - Handling of large/continuous state spaces
    - Memory efficiency (fixed network size vs O(states) table)
    """
    
    def __init__(self, env, 
                 hidden_layers=None,
                 advantage_buffer_size=100000,
                 policy_buffer_size=100000,
                 batch_size=256,
                 train_steps=1000,
                 learning_rate=0.001,
                 device=None,
                 model_path='./deep_cfr_model'):
        """Initialize Deep CFR Agent.
        
        Args:
            env: RLCard environment
            hidden_layers: List of hidden layer sizes for networks
            advantage_buffer_size: Size of advantage sample buffer
            policy_buffer_size: Size of policy sample buffer
            batch_size: Batch size for network training
            train_steps: Number of training steps per iteration
            learning_rate: Learning rate for networks
            device: torch device (cuda/cpu)
            model_path: Path for saving models
        """
        self.use_raw = False
        self.env = env
        self.model_path = model_path
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.hidden_layers = hidden_layers
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # Get dimensions from env
        self.num_actions = env.num_actions
        self.num_players = env.num_players
        
        # State shape - use max across players
        self.state_shape = max(env.state_shape, key=lambda x: np.prod(x))
        
        # Create networks for each player
        self.advantage_nets = []
        self.advantage_optimizers = []
        self.policy_net = None
        self.policy_optimizer = None
        
        for _ in range(self.num_players):
            net = AdvantageNetwork(self.state_shape, self.num_actions, hidden_layers)
            net = net.to(self.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            self.advantage_nets.append(net)
            self.advantage_optimizers.append(optimizer)
        
        # Single policy network (average strategy)
        self.policy_net = PolicyNetwork(self.state_shape, self.num_actions, hidden_layers)
        self.policy_net = self.policy_net.to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffers for training samples
        self.advantage_buffers = [ReservoirBuffer(advantage_buffer_size) 
                                  for _ in range(self.num_players)]
        self.policy_buffer = ReservoirBuffer(policy_buffer_size)
        
        # Training params
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.learning_rate = learning_rate
        
        self.iteration = 0
        
        # Statistics
        self.stats = {
            'nodes_visited': 0,
            'samples_collected': 0,
            'advantage_loss': 0.0,
            'policy_loss': 0.0,
        }

    def train(self):
        """Perform one iteration of Deep CFR training."""
        self.iteration += 1
        self.stats = {'nodes_visited': 0, 'samples_collected': 0, 
                      'advantage_loss': 0.0, 'policy_loss': 0.0}
        
        # Traverse and collect samples for each player
        for player_id in range(self.num_players):
            self.env.reset()
            self._traverse_and_collect(player_id)
        
        # Train networks on collected samples
        if self.iteration >= 2:  # Need some samples first
            self._train_networks()
        
        return self.stats

    def _traverse_and_collect(self, player_id):
        """Traverse game tree collecting advantage and policy samples.
        
        Uses external sampling like MCCFR for efficiency.
        """
        probs = np.ones(self.num_players)
        self._traverse_external(probs, player_id)

    def _traverse_external(self, probs, player_id):
        """External sampling traversal that collects training samples."""
        self.stats['nodes_visited'] += 1
        
        if self.env.is_over():
            return self.env.get_payoffs()
        
        current_player = self.env.get_player_id()
        state = self.env.get_state(current_player)
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        
        # Get current policy via regret matching on predicted advantages
        action_probs = self._get_action_probs(obs, legal_actions, current_player)
        
        if current_player == player_id:
            # Traversing player: explore ALL actions
            action_utilities = {}
            state_utility = np.zeros(self.num_players)
            
            for action in legal_actions:
                action_prob = action_probs[action]
                new_probs = probs.copy()
                new_probs[current_player] *= action_prob
                
                self.env.step(action)
                utility = self._traverse_external(new_probs, player_id)
                self.env.step_back()
                
                state_utility += action_prob * utility
                action_utilities[action] = utility
            
            # Compute advantages (regrets)
            advantages = np.zeros(self.num_actions)
            for action in legal_actions:
                advantages[action] = (action_utilities[action][current_player] - 
                                     state_utility[current_player])
            
            # Store advantage sample
            sample = AdvantageSample(
                state=obs.copy(),
                advantages=advantages.copy(),
                iteration=self.iteration
            )
            self.advantage_buffers[current_player].add(sample)
            
            # Store policy sample
            policy_sample = PolicySample(
                state=obs.copy(),
                policy=action_probs.copy()
            )
            self.policy_buffer.add(policy_sample)
            
            self.stats['samples_collected'] += 1
            
            return state_utility
        else:
            # Opponent: sample ONE action
            action = self._sample_action(action_probs, legal_actions)
            
            new_probs = probs.copy()
            new_probs[current_player] *= action_probs[action]
            
            self.env.step(action)
            utility = self._traverse_external(new_probs, player_id)
            self.env.step_back()
            
            return utility

    def _get_action_probs(self, obs, legal_actions, player_id):
        """Get action probabilities using regret matching on predicted advantages."""
        # Get predicted advantages from network
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            advantages = self.advantage_nets[player_id](state_tensor)
            advantages = advantages.cpu().numpy()[0]
        
        # Apply regret matching
        action_probs = self._regret_matching(advantages, legal_actions)
        return action_probs

    def _regret_matching(self, advantages, legal_actions):
        """Convert advantages to action probabilities via regret matching."""
        action_probs = np.zeros(self.num_actions)
        
        # Only consider legal actions with positive advantages
        positive_sum = 0.0
        for a in legal_actions:
            if advantages[a] > 0:
                positive_sum += advantages[a]
        
        if positive_sum > 0:
            for a in legal_actions:
                action_probs[a] = max(0, advantages[a]) / positive_sum
        else:
            # Uniform over legal actions
            for a in legal_actions:
                action_probs[a] = 1.0 / len(legal_actions)
        
        return action_probs

    def _sample_action(self, action_probs, legal_actions):
        """Sample action from probabilities."""
        legal_probs = np.array([action_probs[a] for a in legal_actions])
        legal_probs = legal_probs / legal_probs.sum()
        idx = np.random.choice(len(legal_actions), p=legal_probs)
        return legal_actions[idx]

    def _train_networks(self):
        """Train advantage and policy networks on collected samples."""
        # Train advantage networks
        for player_id in range(self.num_players):
            if len(self.advantage_buffers[player_id]) >= self.batch_size:
                loss = self._train_advantage_net(player_id)
                self.stats['advantage_loss'] += loss
        
        # Train policy network
        if len(self.policy_buffer) >= self.batch_size:
            loss = self._train_policy_net()
            self.stats['policy_loss'] = loss

    def _train_advantage_net(self, player_id):
        """Train advantage network for a player."""
        net = self.advantage_nets[player_id]
        optimizer = self.advantage_optimizers[player_id]
        buffer = self.advantage_buffers[player_id]
        
        total_loss = 0.0
        num_batches = min(self.train_steps, len(buffer) // self.batch_size)
        
        for _ in range(max(1, num_batches)):
            samples = buffer.sample(self.batch_size)
            
            # Prepare batch with iteration weighting
            states = np.array([s.state for s in samples])
            advantages = np.array([s.advantages for s in samples])
            iterations = np.array([s.iteration for s in samples])
            
            # Weight by iteration (linear CFR)
            weights = iterations / iterations.max()
            
            states_t = torch.FloatTensor(states).to(self.device)
            advantages_t = torch.FloatTensor(advantages).to(self.device)
            weights_t = torch.FloatTensor(weights).to(self.device)
            
            # Forward pass
            predicted = net(states_t)
            
            # Weighted MSE loss
            loss = ((predicted - advantages_t) ** 2).mean(dim=1)
            loss = (loss * weights_t).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(1, num_batches)

    def _train_policy_net(self):
        """Train policy network on collected samples."""
        total_loss = 0.0
        num_batches = min(self.train_steps, len(self.policy_buffer) // self.batch_size)
        
        for _ in range(max(1, num_batches)):
            samples = self.policy_buffer.sample(self.batch_size)
            
            states = np.array([s.state for s in samples])
            policies = np.array([s.policy for s in samples])
            
            states_t = torch.FloatTensor(states).to(self.device)
            policies_t = torch.FloatTensor(policies).to(self.device)
            
            # Forward pass - get log probs
            log_probs = self.policy_net(states_t)
            
            # Cross-entropy loss
            loss = -(policies_t * log_probs).sum(dim=1).mean()
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / max(1, num_batches)

    def eval_step(self, state):
        """Predict action using policy network (for evaluation)."""
        obs = state['obs']
        legal_actions = list(state['legal_actions'].keys())
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            log_probs = self.policy_net(state_tensor)
            probs = torch.exp(log_probs).cpu().numpy()[0]
        
        # Zero out illegal actions and renormalize
        probs = remove_illegal(probs, legal_actions)
        
        action = np.random.choice(len(probs), p=probs)
        
        info = {'probs': {state['raw_legal_actions'][i]: float(probs[legal_actions[i]]) 
                         for i in range(len(legal_actions))}}
        
        return action, info

    def save(self):
        """Save model to disk."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        checkpoint = {
            'iteration': self.iteration,
            'advantage_nets': [net.state_dict() for net in self.advantage_nets],
            'policy_net': self.policy_net.state_dict(),
            'hidden_layers': self.hidden_layers,
            'state_shape': self.state_shape,
            'num_actions': self.num_actions,
        }
        
        torch.save(checkpoint, os.path.join(self.model_path, 'model.pt'))

    def load(self):
        """Load model from disk."""
        path = os.path.join(self.model_path, 'model.pt')
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.iteration = checkpoint['iteration']
        
        for i, state_dict in enumerate(checkpoint['advantage_nets']):
            self.advantage_nets[i].load_state_dict(state_dict)
        
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        
        return True
