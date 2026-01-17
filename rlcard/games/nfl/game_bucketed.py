"""
Simplified NFL Game with Bucketed States

Uses discrete buckets for down, distance, and field position instead of 
continuous values. This dramatically reduces the state space:
- Continuous: 4 × 30 × 100 = 12,000+ states
- Bucketed:   4 × 4 × 5   = 80 states

This makes tabular CFR algorithms (MCCFR) much more efficient.
"""

import numpy as np
from rlcard.games.nfl.game import NFLGame, FORMATION_ACTIONS, DEFENSE_ACTIONS, PLAY_TYPE_ACTIONS, INITIAL_ACTIONS, SPECIAL_TEAMS


# Distance buckets
DISTANCE_BUCKETS = {
    'short':     (1, 3),    # 1-3 yards
    'medium':    (4, 7),    # 4-7 yards  
    'long':      (8, 15),   # 8-15 yards
    'very_long': (16, 99),  # 16+ yards
}

# Field position buckets (5-yard increments = 20 buckets)
FIELD_POSITION_BUCKETS = {
    'own_1_5':    (1, 5),
    'own_6_10':   (6, 10),
    'own_11_15':  (11, 15),
    'own_16_20':  (16, 20),
    'own_21_25':  (21, 25),
    'own_26_30':  (26, 30),
    'own_31_35':  (31, 35),
    'own_36_40':  (36, 40),
    'own_41_45':  (41, 45),
    'own_46_50':  (46, 50),
    'opp_49_45':  (51, 55),
    'opp_44_40':  (56, 60),
    'opp_39_35':  (61, 65),
    'opp_34_30':  (66, 70),
    'opp_29_25':  (71, 75),
    'opp_24_20':  (76, 80),
    'opp_19_15':  (81, 85),
    'opp_14_10':  (86, 90),
    'opp_9_5':    (91, 95),
    'opp_4_1':    (96, 99),
}


def get_distance_bucket(ydstogo):
    """Convert yards to go to bucket index (0-3)."""
    for i, (name, (low, high)) in enumerate(DISTANCE_BUCKETS.items()):
        if low <= ydstogo <= high:
            return i
    return 3  # Default to very_long


def get_field_position_bucket(yardline):
    """Convert yardline to bucket index (0-19)."""
    for i, (name, (low, high)) in enumerate(FIELD_POSITION_BUCKETS.items()):
        if low <= yardline <= high:
            return i
    return 19  # Default to goal line


class NFLGameBucketed(NFLGame):
    """NFL Game variant with bucketed state space.
    
    State representation uses discrete buckets:
    - Down: 1-4 (4 values)
    - Distance bucket: short/medium/long/very_long (4 values)
    - Field position bucket: 5-yard increments (20 values)
    
    Total info sets per phase:
    - Phase 0: 4 × 4 × 20 = 320
    - Phase 1: 320 × 5 formations = 1600  
    - Phase 2: 320 × 5 × 5 defenses = 8000
    """
    
    def __init__(self, allow_step_back=False, single_play=True):
        """Initialize bucketed NFL game.
        
        Args:
            allow_step_back: Support step_back for CFR
            single_play: End game after one play (recommended for CFR)
        """
        # Mark as bucketed BEFORE calling parent init
        self.is_bucketed = True
        # Call parent init - will load B-spline models for special teams
        super().__init__(
            allow_step_back=allow_step_back, 
            use_simple_model=False,  # Use fitted models for special teams
            single_play=single_play
        )
    
    def get_state(self, player_id):
        """Get state with bucketed representation.
        
        Returns state dict with bucketed observations suitable for tabular methods.
        Extends parent with obs_tuple for tabular CFR.
        """
        # Get base state from parent (now includes obs, raw_legal_actions, etc.)
        base_state = super().get_state(player_id)
        
        # Create bucketed observation
        down_bucket = self.down - 1  # 0-3
        distance_bucket = get_distance_bucket(self.ydstogo)  # 0-3
        field_bucket = get_field_position_bucket(self.yardline)  # 0-19
        
        # Get legal actions from base state (now a dict)
        legal_actions = base_state['legal_actions']
        
        # Encode state as tuple for hashability in tabular methods
        if self.phase == 0:
            # Formation phase: just game state
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket)
        elif self.phase == 1:
            # Defense phase: game state + formation
            # pending_formation could be a formation or None (if special teams was used)
            formation_idx = INITIAL_ACTIONS.index(self.pending_formation) if self.pending_formation in INITIAL_ACTIONS else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx)
        else:
            # Play type phase: game state + formation + box count
            formation_idx = INITIAL_ACTIONS.index(self.pending_formation) if self.pending_formation in INITIAL_ACTIONS else 0
            defense_idx = DEFENSE_ACTIONS.index(self.pending_defense_action) if self.pending_defense_action else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx, defense_idx)
        
        # Override obs with bucketed version for neural networks
        obs_array = np.zeros(12, dtype=np.float32)
        obs_array[0] = down_bucket / 3.0
        obs_array[1] = distance_bucket / 3.0
        obs_array[2] = field_bucket / 19.0
        obs_array[11] = self.phase / 2.0
        
        # Start with base state and add bucketed extras
        state = base_state.copy()
        state['obs'] = obs_array
        state['obs_tuple'] = obs_tuple
        state['phase'] = self.phase  # Ensure int phase
        state['down_bucket'] = down_bucket
        state['distance_bucket'] = distance_bucket
        state['field_bucket'] = field_bucket
        
        return state
    
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
        
        Phase 0 (formation):  4 downs × 4 distance × 20 field = 320
        Phase 1 (defense):    320 × 5 formations = 1600
        Phase 2 (play_type):  320 × 5 formations × 5 defenses = 8000
        
        Total: 9920 information sets
        """
        num_field_buckets = 20  # 5-yard increments from 1-99
        base = 4 * 4 * num_field_buckets  # 320
        phase_0 = base
        phase_1 = base * len(FORMATION_ACTIONS)  # 1600
        phase_2 = base * len(FORMATION_ACTIONS) * len(DEFENSE_ACTIONS)  # 8000
        return {
            'phase_0': phase_0,
            'phase_1': phase_1,
            'phase_2': phase_2,
            'total': phase_0 + phase_1 + phase_2,
        }
