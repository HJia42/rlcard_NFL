"""NFL Environment for RLCard"""

import numpy as np

from rlcard.envs.env import Env
from rlcard.games.nfl import Game as NFLGame


class NFLEnv(Env):
    """NFL Play-by-Play Environment for RLCard.
    
    Two-player imperfect information game:
    - Player 0: Offense (selects formation + play type)
    - Player 1: Defense (sees formation, selects box + personnel)
    """
    
    name = 'nfl'
    
    default_game_config = {}
    
    def __init__(self, config):
        """Initialize NFL environment.
        
        Args:
            config: Configuration dict with 'seed', 'allow_step_back', etc.
        """
        self.game = NFLGame(allow_step_back=config.get('allow_step_back', False))
        super().__init__(config)
        
        # State dimensions
        # Offense: [down, ydstogo, yardline] = 3 dims + one-hot formations (optional)
        # Defense: [down, ydstogo, yardline, formation_one_hot] = 3 + 7 = 10 dims
        self.state_shape = [[3], [10]]  # Different for each player
        self.action_shape = [None, None]  # Discrete actions
        
        # Action mappings (for decode)
        self.offense_actions = self.game.offense_actions
        self.defense_actions = self.game.defense_actions
        
        # Formation encoding
        self.formations = ("SHOTGUN", "SINGLEBACK", "I_FORM", "PISTOL", "EMPTY", "JUMBO", "WILDCAT")
        self.formation_to_idx = {f: i for i, f in enumerate(self.formations)}
    
    def _extract_state(self, state):
        """Extract state features for neural network.
        
        Args:
            state: Raw game state dict
            
        Returns:
            Dict with 'obs', 'legal_actions', 'raw_obs', 'raw_legal_actions'
        """
        player_id = state.get('player_id', 0)
        
        # Base features
        down = state['down'] / 4.0  # Normalize
        ydstogo = min(state['ydstogo'], 30) / 30.0
        yardline = state['yardline'] / 100.0
        
        if player_id == 0:
            # Offense state
            obs = np.array([down, ydstogo, yardline], dtype=np.float32)
            legal_actions = self._encode_actions(state['legal_actions'], is_offense=True)
        else:
            # Defense state - includes formation
            formation = state.get('formation', 'UNKNOWN')
            formation_one_hot = self._encode_formation(formation)
            obs = np.array([down, ydstogo, yardline] + formation_one_hot, dtype=np.float32)
            legal_actions = self._encode_actions(state['legal_actions'], is_offense=False)
        
        extracted_state = {
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': state['legal_actions']
        }
        
        return extracted_state
    
    def _encode_formation(self, formation):
        """One-hot encode formation."""
        one_hot = [0] * len(self.formations)
        if formation in self.formation_to_idx:
            one_hot[self.formation_to_idx[formation]] = 1
        return one_hot
    
    def _encode_actions(self, action_ids, is_offense):
        """Create action mask."""
        if is_offense:
            mask = {i: None for i in action_ids}
        else:
            mask = {i: None for i in action_ids}
        return mask
    
    def _decode_action(self, action_id):
        """Decode action ID to game action.
        
        RLCard passes action IDs; we return them as-is since
        the game handles the mapping.
        """
        return action_id
    
    def _get_legal_actions(self):
        """Get legal actions for current player."""
        return self.game.get_legal_actions()
    
    def get_payoffs(self):
        """Get payoffs for all players."""
        return self.game.get_payoffs()
    
    def get_perfect_information(self):
        """Get perfect information state (for debugging)."""
        return {
            'down': self.game.down,
            'ydstogo': self.game.ydstogo,
            'yardline': self.game.yardline,
            'pending_offense_action': self.game.pending_offense_action,
            'current_player': self.game.current_player,
            'is_over': self.game.is_over()
        }
