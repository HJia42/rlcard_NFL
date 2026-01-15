"""
PPO Agent for NFL Play-Calling

Implements Proximal Policy Optimization for two-player self-play training
in the NFL environment. Works with both 'nfl' and 'nfl-bucketed' games.

Key features:
- Actor-Critic architecture
- Self-play training (same agent plays both offense and defense)
- Action masking for legal actions
- Supports variable action spaces per phase
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import deque


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO with action masking."""
    
    def __init__(self, state_dim, action_dim, hidden_dims=[128, 128]):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dims[-1], action_dim)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dims[-1], 1)
        
    def forward(self, state, action_mask=None):
        """Forward pass.
        
        Args:
            state: State tensor [batch, state_dim]
            action_mask: Boolean mask of legal actions [batch, action_dim]
            
        Returns:
            action_probs: Probability distribution over actions
            value: State value estimate
        """
        features = self.shared(state)
        
        # Actor: policy logits
        logits = self.actor(features)
        
        # Apply action mask (set illegal actions to -inf)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, float('-inf'))
        
        action_probs = F.softmax(logits, dim=-1)
        
        # Critic: state value
        value = self.critic(features)
        
        return action_probs, value
    
    def get_action(self, state, action_mask=None, deterministic=False):
        """Sample action from policy.
        
        Args:
            state: State tensor [state_dim]
            action_mask: Boolean mask of legal actions
            deterministic: If True, return argmax action
            
        Returns:
            action: Selected action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_probs, value = self.forward(
            state.unsqueeze(0), 
            action_mask.unsqueeze(0) if action_mask is not None else None
        )
        
        dist = Categorical(action_probs)
        
        if deterministic:
            action = action_probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.squeeze(), value.squeeze()


class PPOAgent:
    """PPO Agent for RLCard NFL environment."""
    
    def __init__(
        self,
        state_shape,
        num_actions,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        n_epochs=4,
        batch_size=64,
        hidden_dims=[128, 128],
        device=None,
    ):
        """Initialize PPO agent.
        
        Args:
            state_shape: Shape of state observation
            num_actions: Maximum number of actions
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Max gradient norm for clipping
            n_epochs: Number of PPO update epochs
            batch_size: Mini-batch size for updates
            hidden_dims: Hidden layer dimensions
            device: PyTorch device
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.num_actions = num_actions
        
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # State dimension
        if isinstance(state_shape, (list, tuple)):
            state_dim = state_shape[0] if len(state_shape) == 1 else np.prod(state_shape)
        else:
            state_dim = state_shape
            
        # Create network
        self.network = ActorCritic(
            state_dim=state_dim, 
            action_dim=num_actions,
            hidden_dims=hidden_dims
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
        
        # Training stats
        self.use_raw = False  # For RLCard compatibility
        
    def step(self, state, deterministic=False):
        """Select action given state (RLCard interface).
        
        Args:
            state: State dictionary from environment
            deterministic: If True, use greedy action
            
        Returns:
            action: Selected action index
        """
        obs = self._process_state(state)
        action_mask = self._get_action_mask(state)
        
        with torch.no_grad():
            action, log_prob, value = self.network.get_action(
                obs, action_mask, deterministic=deterministic
            )
        
        # Store for training
        self.states.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.action_masks.append(action_mask)
        
        return action
    
    def eval_step(self, state):
        """Evaluation step - greedy action (RLCard interface).
        
        Args:
            state: State dictionary from environment
            
        Returns:
            action: Selected action index
            probs: Action probability dictionary
        """
        obs = self._process_state(state)
        action_mask = self._get_action_mask(state)
        
        with torch.no_grad():
            action_probs, _ = self.network(
                obs.unsqueeze(0),
                action_mask.unsqueeze(0) if action_mask is not None else None
            )
        
        action = action_probs.argmax(dim=-1).item()
        probs = {i: p.item() for i, p in enumerate(action_probs.squeeze())}
        
        return action, probs
    
    def batch_eval_step(self, states, deterministic=True):
        """Batched evaluation for multiple states at once.
        
        More efficient than calling eval_step repeatedly when
        evaluating against many opponents or simulating many games.
        
        Args:
            states: List of state dictionaries
            deterministic: If True, return argmax actions
            
        Returns:
            actions: List of action indices
            probs_list: List of probability dicts
        """
        if not states:
            return [], []
        
        # Process all states to tensors
        obs_list = []
        mask_list = []
        
        for state in states:
            obs = self._process_state(state)
            mask = self._get_action_mask(state)
            obs_list.append(obs)
            mask_list.append(mask)
        
        # Stack into batches
        batch_obs = torch.stack(obs_list)
        
        if mask_list[0] is not None:
            batch_masks = torch.stack(mask_list)
        else:
            batch_masks = None
        
        # Single forward pass for all states
        with torch.no_grad():
            action_probs, _ = self.network(batch_obs, batch_masks)
        
        if deterministic:
            actions = action_probs.argmax(dim=-1)
        else:
            from torch.distributions import Categorical
            dist = Categorical(action_probs)
            actions = dist.sample()
        
        actions_list = actions.cpu().tolist()
        probs_list = [
            {i: p for i, p in enumerate(probs)}
            for probs in action_probs.cpu().numpy().tolist()
        ]
        
        return actions_list, probs_list
    
    def feed(self, ts):
        """Feed transition for training (RLCard interface).
        
        Args:
            ts: Tuple of (state, action, reward, next_state, done)
        """
        _, _, reward, _, done = ts
        self.rewards.append(reward)
        self.dones.append(done)
    
    def _process_state(self, state):
        """Convert state dict to tensor."""
        obs = state['obs']
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)
        if isinstance(obs, np.ndarray):
            return torch.FloatTensor(obs).to(self.device)
        if isinstance(obs, list):
            return torch.FloatTensor(obs).to(self.device)
        return torch.FloatTensor([obs]).to(self.device)
    
    def _get_action_mask(self, state):
        """Get action mask from state."""
        legal_actions = state.get('legal_actions', {})
        
        # Handle different legal_actions formats
        if isinstance(legal_actions, dict):
            legal_indices = list(legal_actions.keys())
        elif isinstance(legal_actions, list):
            legal_indices = legal_actions
        else:
            return None
            
        mask = torch.zeros(self.num_actions, dtype=torch.bool, device=self.device)
        for idx in legal_indices:
            if isinstance(idx, int) and 0 <= idx < self.num_actions:
                mask[idx] = True
        
        return mask
    
    def update(self):
        """Perform PPO update using collected rollout data.
        
        Returns:
            dict: Training statistics
        """
        if len(self.rewards) == 0:
            return {}
        
        # Convert to tensors
        states = torch.stack(self.states)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.stack(self.values)
        dones = torch.FloatTensor(self.dones).to(self.device)
        
        # Stack action masks (handle None)
        if self.action_masks[0] is not None:
            action_masks = torch.stack(self.action_masks)
        else:
            action_masks = None
        
        # Compute advantages using GAE
        advantages = self._compute_gae(rewards, values, dones)
        returns = advantages + values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = action_masks[batch_indices] if action_masks is not None else None
                
                # Forward pass
                action_probs, values_new = self.network(batch_states, batch_masks)
                dist = Categorical(action_probs)
                
                log_probs_new = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # PPO clipped objective
                ratio = torch.exp(log_probs_new - batch_old_log_probs.detach())
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values_new.squeeze(-1), batch_returns)
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.value_coef * value_loss 
                    - self.entropy_coef * entropy.mean()
                )
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1
        
        # Clear rollout buffer
        self.clear_buffer()
        
        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return advantages
    
    def clear_buffer(self):
        """Clear rollout buffer."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Saved PPO model to {path}")
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded PPO model from {path}")
