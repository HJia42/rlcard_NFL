"""
Parallel Deep CFR Trainer

Uses multiple actor processes to collect samples in parallel, with a 
centralized learner training networks on GPU. Based on DMC patterns.
"""

import os
import time
import threading
from collections import deque

import numpy as np
import torch
from torch import multiprocessing as mp

import rlcard
from rlcard.agents.deep_cfr_agent import (
    DeepCFRAgent, AdvantageNetwork, PolicyNetwork, 
    ReservoirBuffer, AdvantageSample, PolicySample
)
from rlcard.utils.utils import remove_illegal


def actor_process(
    actor_id,
    env_config,
    advantage_queues,
    policy_queue,
    model_state_dict_queue,
    stop_event,
    num_players,
    num_actions,
    state_shape,
    hidden_layers,
):
    """Actor process that collects samples via tree traversal.
    
    Args:
        actor_id: Unique ID for this actor
        env_config: Config dict for creating environment
        advantage_queues: List of queues for advantage samples (one per player)
        policy_queue: Queue for policy samples
        model_state_dict_queue: Queue to receive updated model weights
        stop_event: Event to signal stopping
        num_players: Number of players in game
        num_actions: Number of actions
        state_shape: Shape of state observations
        hidden_layers: Hidden layer sizes for networks
    """
    # Create local environment
    env = rlcard.make('nfl', config=env_config)
    
    # Create local networks for inference (CPU)
    advantage_nets = []
    for _ in range(num_players):
        net = AdvantageNetwork(state_shape, num_actions, hidden_layers)
        net.eval()
        advantage_nets.append(net)
    
    iterations = 0
    
    while not stop_event.is_set():
        # Check for weight updates
        try:
            while not model_state_dict_queue.empty():
                state_dicts = model_state_dict_queue.get_nowait()
                for i, sd in enumerate(state_dicts):
                    advantage_nets[i].load_state_dict(sd)
        except:
            pass
        
        iterations += 1
        
        # Traverse for each player
        for player_id in range(num_players):
            env.reset()
            _traverse_external(
                env, player_id, advantage_nets, 
                advantage_queues, policy_queue, 
                num_actions, iterations
            )


def _traverse_external(env, player_id, advantage_nets, 
                       advantage_queues, policy_queue, 
                       num_actions, iteration):
    """External sampling traversal for actor."""
    
    if env.is_over():
        return env.get_payoffs()
    
    current_player = env.get_player_id()
    state = env.get_state(current_player)
    obs = state['obs']
    legal_actions = list(state['legal_actions'].keys())
    
    # Get action probs via regret matching
    with torch.no_grad():
        state_t = torch.FloatTensor(obs).unsqueeze(0)
        advantages = advantage_nets[current_player](state_t).numpy()[0]
    
    action_probs = _regret_matching(advantages, legal_actions, num_actions)
    
    if current_player == player_id:
        # Explore all actions
        action_utilities = {}
        state_utility = np.zeros(2)
        
        for action in legal_actions:
            env.step(action)
            utility = _traverse_external(
                env, player_id, advantage_nets,
                advantage_queues, policy_queue, 
                num_actions, iteration
            )
            env.step_back()
            
            state_utility += action_probs[action] * utility
            action_utilities[action] = utility
        
        # Compute advantages
        adv = np.zeros(num_actions)
        for a in legal_actions:
            adv[a] = action_utilities[a][current_player] - state_utility[current_player]
        
        # Send samples to queues
        sample = AdvantageSample(obs.copy(), adv.copy(), iteration)
        try:
            advantage_queues[current_player].put_nowait(sample)
        except:
            pass  # Queue full, skip
        
        policy_sample = PolicySample(obs.copy(), action_probs.copy())
        try:
            policy_queue.put_nowait(policy_sample)
        except:
            pass
        
        return state_utility
    else:
        # Sample opponent action
        legal_probs = np.array([action_probs[a] for a in legal_actions])
        legal_probs = legal_probs / (legal_probs.sum() + 1e-8)
        action = legal_actions[np.random.choice(len(legal_actions), p=legal_probs)]
        
        env.step(action)
        utility = _traverse_external(
            env, player_id, advantage_nets,
            advantage_queues, policy_queue, 
            num_actions, iteration
        )
        env.step_back()
        
        return utility


def _regret_matching(advantages, legal_actions, num_actions):
    """Convert advantages to action probabilities."""
    action_probs = np.zeros(num_actions)
    positive_sum = sum(max(0, advantages[a]) for a in legal_actions)
    
    if positive_sum > 0:
        for a in legal_actions:
            action_probs[a] = max(0, advantages[a]) / positive_sum
    else:
        for a in legal_actions:
            action_probs[a] = 1.0 / len(legal_actions)
    
    return action_probs


class ParallelDeepCFRTrainer:
    """Parallel Deep CFR trainer with multiple actors and GPU learner."""
    
    def __init__(self,
                 env_config,
                 num_actors=4,
                 hidden_layers=None,
                 batch_size=256,
                 train_steps=100,
                 learning_rate=0.001,
                 advantage_buffer_size=100000,
                 policy_buffer_size=100000,
                 cuda=True,
                 model_path='./models/parallel_deep_cfr'):
        """Initialize parallel trainer.
        
        Args:
            env_config: Config for creating NFL environment
            num_actors: Number of parallel actor processes
            hidden_layers: Network hidden layer sizes
            batch_size: Training batch size
            train_steps: Training steps per iteration
            learning_rate: Learning rate
            advantage_buffer_size: Size of advantage sample buffer
            policy_buffer_size: Size of policy sample buffer
            cuda: Whether to use GPU for training
            model_path: Path for saving models
        """
        self.env_config = env_config
        self.num_actors = num_actors
        self.model_path = model_path
        
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.hidden_layers = hidden_layers
        
        # Create env to get dimensions
        env = rlcard.make('nfl', config=env_config)
        self.num_actions = env.num_actions
        self.num_players = env.num_players
        self.state_shape = max(env.state_shape, key=lambda x: np.prod(x))
        
        # Device
        if cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        # Create learner networks (GPU)
        self.advantage_nets = []
        self.advantage_optimizers = []
        for _ in range(self.num_players):
            net = AdvantageNetwork(self.state_shape, self.num_actions, hidden_layers)
            net = net.to(self.device)
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            self.advantage_nets.append(net)
            self.advantage_optimizers.append(optimizer)
        
        self.policy_net = PolicyNetwork(self.state_shape, self.num_actions, hidden_layers)
        self.policy_net = self.policy_net.to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Buffers
        self.advantage_buffers = [ReservoirBuffer(advantage_buffer_size) 
                                  for _ in range(self.num_players)]
        self.policy_buffer = ReservoirBuffer(policy_buffer_size)
        
        # Training params
        self.batch_size = batch_size
        self.train_steps = train_steps
        
        # Multiprocessing
        self.actors = []
        self.advantage_queues = []
        self.policy_queue = None
        self.model_state_queues = []
        self.stop_event = None
        
        self.iteration = 0
    
    def start_actors(self):
        """Start actor processes."""
        mp.set_start_method('spawn', force=True)
        
        self.stop_event = mp.Event()
        self.advantage_queues = [mp.Queue(maxsize=10000) for _ in range(self.num_players)]
        self.policy_queue = mp.Queue(maxsize=10000)
        self.model_state_queues = [mp.Queue(maxsize=2) for _ in range(self.num_actors)]
        
        for i in range(self.num_actors):
            p = mp.Process(
                target=actor_process,
                args=(
                    i,
                    self.env_config,
                    self.advantage_queues,
                    self.policy_queue,
                    self.model_state_queues[i],
                    self.stop_event,
                    self.num_players,
                    self.num_actions,
                    self.state_shape,
                    self.hidden_layers,
                )
            )
            p.start()
            self.actors.append(p)
        
        print(f"Started {self.num_actors} actor processes")
    
    def stop_actors(self):
        """Stop actor processes."""
        self.stop_event.set()
        for p in self.actors:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.actors = []
    
    def broadcast_weights(self):
        """Send updated weights to actors."""
        state_dicts = [net.cpu().state_dict() for net in self.advantage_nets]
        # Move nets back to device
        for net in self.advantage_nets:
            net.to(self.device)
        
        for q in self.model_state_queues:
            try:
                # Clear old weights
                while not q.empty():
                    q.get_nowait()
                q.put_nowait(state_dicts)
            except:
                pass
    
    def collect_samples(self):
        """Collect samples from actor queues."""
        samples_collected = 0
        
        for player_id in range(self.num_players):
            while not self.advantage_queues[player_id].empty():
                try:
                    sample = self.advantage_queues[player_id].get_nowait()
                    self.advantage_buffers[player_id].add(sample)
                    samples_collected += 1
                except:
                    break
        
        while not self.policy_queue.empty():
            try:
                sample = self.policy_queue.get_nowait()
                self.policy_buffer.add(sample)
            except:
                break
        
        return samples_collected
    
    def train_step(self):
        """Train networks on collected samples."""
        self.iteration += 1
        
        # Collect samples from actors
        samples = self.collect_samples()
        
        # Train advantage networks
        adv_loss = 0.0
        for player_id in range(self.num_players):
            if len(self.advantage_buffers[player_id]) >= self.batch_size:
                adv_loss += self._train_advantage_net(player_id)
        
        # Train policy network
        pol_loss = 0.0
        if len(self.policy_buffer) >= self.batch_size:
            pol_loss = self._train_policy_net()
        
        # Broadcast updated weights
        if self.iteration % 10 == 0:
            self.broadcast_weights()
        
        return {
            'samples': samples,
            'adv_loss': adv_loss,
            'pol_loss': pol_loss,
            'adv_buffer': [len(b) for b in self.advantage_buffers],
            'pol_buffer': len(self.policy_buffer),
        }
    
    def _train_advantage_net(self, player_id):
        """Train advantage network."""
        net = self.advantage_nets[player_id]
        optimizer = self.advantage_optimizers[player_id]
        buffer = self.advantage_buffers[player_id]
        
        samples = buffer.sample(self.batch_size)
        
        states = np.array([s.state for s in samples])
        advantages = np.array([s.advantages for s in samples])
        iterations = np.array([s.iteration for s in samples])
        weights = iterations / (iterations.max() + 1)
        
        states_t = torch.FloatTensor(states).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        weights_t = torch.FloatTensor(weights).to(self.device)
        
        predicted = net(states_t)
        loss = ((predicted - advantages_t) ** 2).mean(dim=1)
        loss = (loss * weights_t).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def _train_policy_net(self):
        """Train policy network."""
        samples = self.policy_buffer.sample(self.batch_size)
        
        states = np.array([s.state for s in samples])
        policies = np.array([s.policy for s in samples])
        
        states_t = torch.FloatTensor(states).to(self.device)
        policies_t = torch.FloatTensor(policies).to(self.device)
        
        log_probs = self.policy_net(states_t)
        loss = -(policies_t * log_probs).sum(dim=1).mean()
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return loss.item()
    
    def get_agent(self):
        """Get agent for evaluation."""
        # Create a DeepCFRAgent with our trained networks
        env = rlcard.make('nfl', config=self.env_config)
        agent = DeepCFRAgent(env, hidden_layers=self.hidden_layers, device=self.device)
        
        # Copy network weights
        for i in range(self.num_players):
            agent.advantage_nets[i].load_state_dict(self.advantage_nets[i].state_dict())
        agent.policy_net.load_state_dict(self.policy_net.state_dict())
        
        return agent
    
    def save(self):
        """Save model."""
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        checkpoint = {
            'iteration': self.iteration,
            'advantage_nets': [net.state_dict() for net in self.advantage_nets],
            'policy_net': self.policy_net.state_dict(),
            'hidden_layers': self.hidden_layers,
        }
        torch.save(checkpoint, os.path.join(self.model_path, 'model.pt'))
    
    def load(self):
        """Load model."""
        path = os.path.join(self.model_path, 'model.pt')
        if not os.path.exists(path):
            return False
        
        checkpoint = torch.load(path, map_location=self.device)
        self.iteration = checkpoint['iteration']
        
        for i, sd in enumerate(checkpoint['advantage_nets']):
            self.advantage_nets[i].load_state_dict(sd)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        
        return True
