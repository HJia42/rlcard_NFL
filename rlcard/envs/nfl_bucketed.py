"""
Bucketed NFL Environment

Environment wrapper for the bucketed NFL game variant.
Uses discrete state buckets for efficient tabular CFR.
"""

import numpy as np
from rlcard.envs.env import Env
from rlcard.games.nfl.game_bucketed import NFLGameBucketed, FORMATION_ACTIONS


class NFLBucketedEnv(Env):
    """RLCard environment for bucketed NFL game."""
    
    name = 'nfl-bucketed'
    
    default_game_config = {}
    
    def __init__(self, config):
        """Initialize bucketed NFL environment."""
        self.game = NFLGameBucketed(
            allow_step_back=config.get('allow_step_back', False),
            single_play=config.get('single_play', True),  # Default to single play for CFR
        )
        super().__init__(config)
        
        # Bucketed observations are smaller
        # 3 dims: [down_bucket, distance_bucket, field_bucket] 
        # Padded to 11 for consistency with standard NFL env
        self.state_shape = [[11], [11]]
        self.action_shape = [None, None]
        
        # Encoding mappings
        self.formations = FORMATION_ACTIONS
    
    def _extract_state(self, state):
        """Extract state dict for agents."""
        extracted = {
            'obs': state['obs'],
            'legal_actions': {i: None for i in state['legal_actions']},
            'raw_obs': state,
            'raw_legal_actions': state['raw_legal_actions'],
        }
        # Include obs_tuple if present (for tabular methods)
        if 'obs_tuple' in state:
            extracted['obs_tuple'] = state['obs_tuple']
        return extracted
    
    def get_payoffs(self):
        """Return the EPA payoffs."""
        return self.game.get_payoffs()
    
    def _decode_action(self, action_id):
        """Decode action ID to game action."""
        return action_id
    
    def _get_legal_actions(self):
        """Get legal actions as dict."""
        return {a: None for a in self.game.get_legal_actions()}
    
    def get_perfect_information(self):
        """Get current game state info."""
        return {
            'down': self.game.down,
            'ydstogo': self.game.ydstogo,
            'yardline': self.game.yardline,
            'phase': self.game.phase,
            'is_over': self.game.is_over(),
        }
