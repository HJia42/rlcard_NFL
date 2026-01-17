"""
NFL IIG with Bucketed States

Combines the IIG (imperfect information) game structure with
bucketed state representation for tabular methods like CFR.

3-Phase Structure:
- Phase 0: Offense picks formation (or PUNT/FG)
- Phase 1: Offense picks play type (pass/rush) - hidden from defense
- Phase 2: Defense sees formation, picks box count
"""

import numpy as np
from rlcard.games.nfl.game_iig import NFLGameIIG
from rlcard.games.nfl.game_bucketed import (
    get_distance_bucket,
    get_field_position_bucket,
)
from rlcard.games.nfl.game import (
    DEFENSE_ACTIONS, 
    FORMATION_ACTIONS, 
    INITIAL_ACTIONS,
    PLAY_TYPE_ACTIONS,
)


class NFLGameIIGBucketed(NFLGameIIG):
    """NFL IIG with bucketed state space for tabular CFR.
    
    State representation uses discrete buckets:
    - Down: 1-4 (4 values)
    - Distance: short/medium/long/very_long (4 values)
    - Field position: 20 buckets (5-yard increments)
    
    Information sets:
    - Phase 0: 4 × 4 × 20 = 320 (offense picks formation)
    - Phase 1: 320 × 7 formations = 2240 (offense picks play type)
    - Phase 2: 320 × 5 formations = 1600 (defense, sees formation only)
    """
    
    def __init__(self, allow_step_back=False, single_play=True, 
                 use_cached_model=True, seed=None):
        """Initialize bucketed IIG NFL game."""
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
        
        down_bucket = self.down - 1
        distance_bucket = get_distance_bucket(self.ydstogo)
        field_bucket = get_field_position_bucket(self.yardline)
        
        # Build obs_tuple for tabular methods
        if self.phase == 0:
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket)
        elif self.phase == 1:
            # Formation selected, choosing play type
            formation_idx = INITIAL_ACTIONS.index(self.pending_formation) if self.pending_formation in INITIAL_ACTIONS else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx)
        elif self.phase == 2:
            # Defense sees formation only (NOT play type or formation index in action space)
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation) if self.pending_formation in FORMATION_ACTIONS else 0
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket, formation_idx)
        else:
            obs_tuple = (self.phase, down_bucket, distance_bucket, field_bucket)
        
        # Bucketed observation array
        obs_array = np.zeros(12, dtype=np.float32)
        obs_array[0] = down_bucket / 3.0
        obs_array[1] = distance_bucket / 3.0
        obs_array[2] = field_bucket / 19.0
        obs_array[11] = self.phase / 2.0
        
        # Formation encoding for phases 1-2
        if self.phase >= 1 and self.pending_formation in FORMATION_ACTIONS:
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation)
            obs_array[3 + formation_idx] = 1.0
        
        state = base_state.copy()
        state['obs'] = obs_array
        state['obs_tuple'] = obs_tuple
        state['down_bucket'] = down_bucket
        state['distance_bucket'] = distance_bucket
        state['field_bucket'] = field_bucket
        
        return state
    
    def get_info_set_id(self, player_id):
        """Get unique info set ID for tabular methods."""
        state = self.get_state(player_id)
        return str(state['obs_tuple'])
    
    @staticmethod
    def count_info_sets():
        """Return theoretical number of information sets.
        
        Phase 0 (formation): 4 downs × 4 distance × 20 field = 320
        Phase 1 (play_type): 320 × 7 initial actions = 2240
        Phase 2 (defense):   320 × 5 formations = 1600
        
        Total: 4160 information sets
        """
        base = 4 * 4 * 20  # 320
        phase_0 = base
        phase_1 = base * len(INITIAL_ACTIONS)  # 2240
        phase_2 = base * len(FORMATION_ACTIONS)  # 1600 (defense sees formations only)
        return {
            'phase_0': phase_0,
            'phase_1': phase_1,
            'phase_2': phase_2,
            'total': phase_0 + phase_1 + phase_2,
        }
