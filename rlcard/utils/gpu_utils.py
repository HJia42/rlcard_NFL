"""
Device Selection and GPU Utilities for RLCard NFL Agents.

Provides auto-detection of optimal device based on:
- GPU availability
- Batch size
- Network size

Usage:
    from rlcard.utils.gpu_utils import select_device, get_device_info
    
    device = select_device(batch_size=64)  # Returns 'cuda' or 'cpu'
"""

import torch
from typing import Optional


def select_device(
    batch_size: int = 1,
    prefer_gpu: bool = True,
    force_device: Optional[str] = None
) -> torch.device:
    """Select optimal PyTorch device.
    
    Args:
        batch_size: Expected batch size (GPU only beneficial for large batches)
        prefer_gpu: If True, use GPU when available and beneficial
        force_device: Force specific device ('cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        torch.device for model placement
    """
    if force_device is not None:
        if force_device == 'auto':
            pass  # Fall through to auto-detection
        else:
            return torch.device(force_device)
    
    if not prefer_gpu:
        return torch.device('cpu')
    
    if not torch.cuda.is_available():
        return torch.device('cpu')
    
    # GPU beneficial for batch_size >= 16 typically
    # For small batches, CPU may be faster due to transfer overhead
    if batch_size < 16:
        return torch.device('cpu')
    
    return torch.device('cuda')


def get_device_info() -> dict:
    """Get information about available devices.
    
    Returns:
        Dict with device information
    """
    info = {
        'cuda_available': torch.cuda.is_available(),
        'cpu_count': torch.get_num_threads(),
        'recommended': 'cpu',
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total': torch.cuda.get_device_properties(0).total_memory // (1024**3),
            'recommended': 'cuda',
        })
    
    return info


def move_to_device(tensor_or_dict, device: torch.device):
    """Move tensor or dict of tensors to device.
    
    Args:
        tensor_or_dict: Tensor, ndarray, or dict containing tensors
        device: Target device
        
    Returns:
        Same structure on target device
    """
    import numpy as np
    
    if isinstance(tensor_or_dict, torch.Tensor):
        return tensor_or_dict.to(device)
    
    if isinstance(tensor_or_dict, np.ndarray):
        return torch.from_numpy(tensor_or_dict).float().to(device)
    
    if isinstance(tensor_or_dict, dict):
        return {k: move_to_device(v, device) for k, v in tensor_or_dict.items()}
    
    if isinstance(tensor_or_dict, (list, tuple)):
        return type(tensor_or_dict)(move_to_device(v, device) for v in tensor_or_dict)
    
    return tensor_or_dict


class BatchedInference:
    """Batched inference wrapper for neural network agents.
    
    Collects states and performs batched forward passes for efficiency.
    
    Usage:
        batcher = BatchedInference(agent.network, device, batch_size=32)
        
        # Collect states
        for state in states:
            batcher.add(state, action_mask)
        
        # Get all results at once
        actions, probs = batcher.get_actions()
    """
    
    def __init__(self, network, device, batch_size=32):
        self.network = network
        self.device = device
        self.batch_size = batch_size
        self.states = []
        self.masks = []
    
    def add(self, state, action_mask=None):
        """Add state to batch."""
        if isinstance(state, dict):
            state = state.get('obs', state)
        
        self.states.append(move_to_device(state, self.device))
        self.masks.append(
            move_to_device(action_mask, self.device) 
            if action_mask is not None else None
        )
    
    def get_actions(self, deterministic=False):
        """Process all collected states and return actions.
        
        Returns:
            Tuple of (actions, action_probs_list)
        """
        if not self.states:
            return [], []
        
        # Stack into batch
        batch_states = torch.stack(self.states)
        
        if self.masks[0] is not None:
            batch_masks = torch.stack(self.masks)
        else:
            batch_masks = None
        
        # Forward pass
        with torch.no_grad():
            action_probs, _ = self.network(batch_states, batch_masks)
        
        if deterministic:
            actions = action_probs.argmax(dim=-1)
        else:
            from torch.distributions import Categorical
            dist = Categorical(action_probs)
            actions = dist.sample()
        
        actions_list = actions.cpu().tolist()
        probs_list = action_probs.cpu().numpy().tolist()
        
        # Clear buffer
        self.states = []
        self.masks = []
        
        return actions_list, probs_list
    
    def __len__(self):
        return len(self.states)
