"""
NFL IIG Bucketed Environment

Environment wrapper for the bucketed IIG NFL game variant.
Uses discrete state buckets for efficient tabular CFR.
"""

import numpy as np
from rlcard.envs.env import Env
from rlcard.games.nfl.game_iig_bucketed import NFLGameIIGBucketed, IIG_ACTION_NAMES


class NFLIIGBucketedEnv(Env):
    """RLCard environment for bucketed IIG NFL game."""
    
    name = 'nfl-iig-bucketed'
    
    default_game_config = {}
    
    def __init__(self, config):
        """Initialize bucketed IIG NFL environment."""
        self.game = NFLGameIIGBucketed(
            allow_step_back=config.get('allow_step_back', False),
            single_play=config.get('single_play', True),
            use_cached_model=config.get('use_cached_model', True),
        )
        super().__init__(config)
        
        # State shape: 12 dimensions (padded for consistency)
        self.state_shape = [[12], [12]]
        self.action_shape = [None, None]
        
        self.action_names = IIG_ACTION_NAMES
    
    def _extract_state(self, state):
        """Extract state dict for agents."""
        extracted = {
            'obs': state['obs'],
            'legal_actions': {i: None for i in state['legal_actions']},
            'raw_obs': state,
            'raw_legal_actions': state.get('raw_legal_actions', []),
        }
        if 'obs_tuple' in state:
            extracted['obs_tuple'] = state['obs_tuple']
        return extracted
    
    def get_payoffs(self):
        """Return the EPA payoffs."""
        return np.array(self.game.payoffs) if hasattr(self.game, 'payoffs') else np.zeros(2)
    
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
