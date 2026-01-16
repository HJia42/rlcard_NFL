"""
NFL IIG with Bucketed States

Combines the IIG (imperfect information) game structure with
bucketed state representation for tabular methods like CFR.

Offense commits to (Formation, PlayType) before defense responds.
"""

import numpy as np
from rlcard.games.nfl.game_iig import (
    NFLGameIIG,
    IIG_OFFENSE_ACTIONS,
    IIG_ACTION_NAMES,
    decode_iig_action,
)
from rlcard.games.nfl.game_bucketed import (
    get_distance_bucket,
    get_field_position_bucket,
    DISTANCE_BUCKETS,
    FIELD_POSITION_BUCKETS,
)
from rlcard.games.nfl.game import DEFENSE_ACTIONS, FORMATION_ACTIONS


class NFLGameIIGBucketed(NFLGameIIG):
    """NFL IIG with bucketed state space for tabular CFR.
    
    State representation uses discrete buckets:
    - Down: 1-4 (4 values)
    - Distance: short/medium/long/very_long (4 values)
    - Field position: 20 buckets (5-yard increments)
    
    Action space:
    - Offense: 12 actions (5 formations × 2 play types + PUNT + FG)
    - Defense: 5 actions (box counts 4-8)
    
    Information sets:
    - Phase 0: 4 × 4 × 20 = 320 (offense)
    - Phase 1: 320 × 5 formations = 1600 (defense sees formation only, NOT play type)
    """
    
    def __init__(self, allow_step_back=False, single_play=True, 
                 use_cached_model=True, seed=None):
        """Initialize bucketed IIG NFL game."""
        # Mark as bucketed BEFORE calling parent init
        self.is_bucketed = True
        super().__init__(
            allow_step_back=allow_step_back,
            single_play=single_play,
            use_cached_model=use_cached_model,
            seed=seed,
        )
    
    def get_state(self, player_id):
        """Get state with bucketed representation."""
        base_state = super().get_state(player_id)
        
        # Create bucket indices
        down_bucket = self.down - 1  # 0-3
        distance_bucket = get_distance_bucket(self.ydstogo)  # 0-3
        field_bucket = get_field_position_bucket(self.yardline)  # 0-19
        
        # Bucketed tuple for tabular methods
        if self.phase == 0:
            # Offense phase: just game state
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket)
        elif self.phase == 1:
            # Defense phase: game state + visible formation (NOT play type)
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation) if self.pending_formation in FORMATION_ACTIONS else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx)
        else:
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket)
        
        # Neural network observation (12 dims)
        obs_array = np.zeros(12, dtype=np.float32)
        obs_array[0] = down_bucket / 3.0
        obs_array[1] = distance_bucket / 3.0
        obs_array[2] = field_bucket / 19.0
        
        # Formation encoding (if visible)
        if self.phase == 1 and self.pending_formation in FORMATION_ACTIONS:
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation)
            obs_array[3 + formation_idx] = 1.0
        
        # Phase encoding
        obs_array[11] = self.phase / 2.0
        
        return {
            'obs': obs_array,
            'obs_tuple': obs_tuple,
            'legal_actions': base_state['legal_actions'],
            'raw_legal_actions': base_state.get('raw_legal_actions', []),
            'player_id': player_id,
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': self.phase,
            'down_bucket': down_bucket,
            'distance_bucket': distance_bucket,
            'field_bucket': field_bucket,
        }
    
    def get_info_set_id(self, player_id):
        """Get unique info set ID for tabular methods."""
        state = self.get_state(player_id)
        return str(state['obs_tuple'])
    
    @staticmethod
    def count_info_sets():
        """Return theoretical number of information sets.
        
        Phase 0 (offense): 4 downs × 4 distance × 20 field = 320
        Phase 1 (defense): 320 × 5 formations = 1600
                          (Defense sees formation, NOT play type)
        
        Total: 1920 information sets
        
        Note: Smaller than audible game because defense
        doesn't need to consider 2 play types per formation.
        """
        base = 4 * 4 * 20  # 320
        phase_0 = base
        phase_1 = base * len(FORMATION_ACTIONS)  # 1600
        return {
            'phase_0': phase_0,
            'phase_1': phase_1,
            'total': phase_0 + phase_1,
        }


# Exports
INITIAL_ACTIONS = IIG_ACTION_NAMES
