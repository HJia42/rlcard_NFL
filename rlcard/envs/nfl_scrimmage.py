
import numpy as np
from rlcard.envs.env import Env
from rlcard.games.nfl.game_scrimmage import NFLGameScrimmage

class NFLScrimmageEnv(Env):
    """
    RLCard environment for PERFECT INFORMATION scrimmage-only NFL game.
    
    3-Phase Standard Structure:
      Phase 0: Offense picks Formation.
      Phase 1: Defense picks Box Count (sees Formation).
      Phase 2: Offense picks Play Type (sees Box Count).
    """
    
    name = 'nfl-scrimmage'
    
    def __init__(self, config):
        """Initialize scrimmage-only Perfect Information NFL environment."""
        self.game = NFLGameScrimmage(
            allow_step_back=config.get('allow_step_back', False),
            single_play=config.get('single_play', True),
            use_cached_model=config.get('use_cached_model', False),
            seed=config.get('seed', None)
        )
        super().__init__(config)
        
        # State shape: 12 dimensions (Phase encoding + game state)
        self.state_shape = [[12], [12]]
        self.action_shape = [None, None]
    
    def _extract_state(self, state):
        """Extract state dict for agents."""
        # For Perfect Info, observations don't need masking.
        # But we maintain the standard structure.
        legal_actions = state['legal_actions']
        if not isinstance(legal_actions, dict):
            legal_actions = {i: None for i in legal_actions}
            
        return {
            'obs': state['obs'],
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': state.get('raw_legal_actions', []),
        }

    def get_payoffs(self):
        """Return the EPA payoffs."""
        return np.array(self.game.payoffs) if hasattr(self.game, 'payoffs') else np.zeros(2)

    def _decode_action(self, action_id):
        return action_id

    def _get_legal_actions(self):
        return {a: None for a in self.game.get_legal_actions()}
