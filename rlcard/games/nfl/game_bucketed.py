"""
Simplified NFL Game with Bucketed States

Uses discrete buckets for down, distance, and field position instead of 
continuous values. This dramatically reduces the state space:
- Continuous: 4 × 30 × 100 = 12,000+ states
- Bucketed:   4 × 4 × 5   = 80 states

This makes tabular CFR algorithms (MCCFR) much more efficient.
"""

import numpy as np
from rlcard.games.nfl.game import NFLGame, FORMATION_ACTIONS, DEFENSE_ACTIONS, PLAY_TYPE_ACTIONS


# Distance buckets
DISTANCE_BUCKETS = {
    'short':     (1, 3),    # 1-3 yards
    'medium':    (4, 7),    # 4-7 yards  
    'long':      (8, 15),   # 8-15 yards
    'very_long': (16, 99),  # 16+ yards
}

# Field position buckets
FIELD_POSITION_BUCKETS = {
    'own_deep':      (1, 20),    # Deep in own territory
    'own_side':      (21, 40),   # Own side of field
    'midfield':      (41, 60),   # Around midfield
    'opp_side':      (61, 80),   # Opponent's side
    'red_zone':      (81, 99),   # Red zone (scoring territory)
}


def get_distance_bucket(ydstogo):
    """Convert yards to go to bucket index (0-3)."""
    for i, (name, (low, high)) in enumerate(DISTANCE_BUCKETS.items()):
        if low <= ydstogo <= high:
            return i
    return 3  # Default to very_long


def get_field_position_bucket(yardline):
    """Convert yardline to bucket index (0-4)."""
    for i, (name, (low, high)) in enumerate(FIELD_POSITION_BUCKETS.items()):
        if low <= yardline <= high:
            return i
    return 4  # Default to red_zone


class NFLGameBucketed(NFLGame):
    """NFL Game variant with bucketed state space.
    
    State representation uses discrete buckets:
    - Down: 1-4 (4 values)
    - Distance bucket: short/medium/long/very_long (4 values)
    - Field position bucket: own_deep/own_side/midfield/opp_side/red_zone (5 values)
    
    Total info sets: 4 × 4 × 5 = 80 (per phase)
    """
    
    def __init__(self, allow_step_back=False, single_play=True):
        """Initialize bucketed NFL game.
        
        Args:
            allow_step_back: Support step_back for CFR
            single_play: End game after one play (recommended for CFR)
        """
        # Call parent init with simple model (no data needed for bucketed version)
        super().__init__(
            allow_step_back=allow_step_back, 
            use_simple_model=True,
            single_play=single_play
        )
    
    def get_state(self, player_id):
        """Get state with bucketed representation.
        
        Returns state dict with bucketed observations suitable for tabular methods.
        """
        # Get base state from parent
        base_state = super().get_state(player_id)
        
        # Create bucketed observation
        down_bucket = self.down - 1  # 0-3
        distance_bucket = get_distance_bucket(self.ydstogo)  # 0-3
        field_bucket = get_field_position_bucket(self.yardline)  # 0-4
        
        # Get legal actions from base state
        legal_actions = base_state['legal_actions']
        
        # Construct raw_legal_actions based on phase
        if self.phase == 0:
            raw_legal_actions = [FORMATION_ACTIONS[i] for i in legal_actions]
        elif self.phase == 1:
            raw_legal_actions = [DEFENSE_ACTIONS[i] for i in legal_actions]
        else:
            raw_legal_actions = [PLAY_TYPE_ACTIONS[i] for i in legal_actions]
        
        # Encode state as tuple for hashability in tabular methods
        if self.phase == 0:
            # Formation phase: just game state
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket)
        elif self.phase == 1:
            # Defense phase: game state + formation
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation) if self.pending_formation else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx)
        else:
            # Play type phase: game state + formation + box count
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation) if self.pending_formation else 0
            defense_idx = DEFENSE_ACTIONS.index(self.pending_defense_action) if self.pending_defense_action else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx, defense_idx)
        
        # Also create array representation for neural networks
        obs_array = np.array([
            down_bucket / 3.0,          # Normalize to 0-1
            distance_bucket / 3.0,
            field_bucket / 4.0,
        ], dtype=np.float32)
        
        # Pad to consistent size
        full_obs = np.zeros(11, dtype=np.float32)
        full_obs[:len(obs_array)] = obs_array
        
        return {
            'obs': full_obs,           # Array for neural networks
            'obs_tuple': obs_tuple,     # Tuple for tabular methods
            'legal_actions': legal_actions,
            'raw_legal_actions': raw_legal_actions,
            'player_id': player_id,
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': base_state['phase'],
            # Bucket info for debugging
            'down_bucket': down_bucket,
            'distance_bucket': distance_bucket,
            'field_bucket': field_bucket,
        }
    
    def get_info_set_id(self, player_id):
        """Get unique info set ID for current state.
        
        Returns a string that uniquely identifies the information set.
        Useful for tabular CFR methods.
        """
        state = self.get_state(player_id)
        return str(state['obs_tuple'])
    
    @staticmethod
    def count_info_sets():
        """Return theoretical number of information sets.
        
        Phase 0 (formation):  4 downs × 4 distance × 5 field = 80
        Phase 1 (defense):    80 × 7 formations = 560
        Phase 2 (play_type):  80 × 7 formations × 5 defenses = 2800
        
        Total: 3440 information sets
        """
        base = 4 * 4 * 5  # 80
        phase_0 = base
        phase_1 = base * len(FORMATION_ACTIONS)  # 560
        phase_2 = base * len(FORMATION_ACTIONS) * len(DEFENSE_ACTIONS)  # 2800
        return {
            'phase_0': phase_0,
            'phase_1': phase_1,
            'phase_2': phase_2,
            'total': phase_0 + phase_1 + phase_2,
        }
