"""NFL Environment for RLCard"""

import numpy as np

from rlcard.envs.env import Env
from rlcard.games.nfl import Game as NFLGame


class NFLEnv(Env):
    """NFL Play-by-Play Environment for RLCard.
    
    Three-phase game per play:
    - Phase 0 (Player 0): Offense picks formation
    - Phase 1 (Player 1): Defense picks box count (sees formation)
    - Phase 2 (Player 0): Offense picks pass/rush (sees box count!)
    """
    
    name = 'nfl'
    
    default_game_config = {}
    
    def __init__(self, config):
        """Initialize NFL environment."""
        self.game = NFLGame(
            allow_step_back=config.get('allow_step_back', False),
            single_play=config.get('single_play', False),
            use_distribution_model=config.get('use_distribution_model', False),
            use_cached_model=config.get('use_cached_model', False),
        )
        super().__init__(config)
        
        # Custom starting state (for targeted training)
        self.start_down = config.get('start_down', None)
        self.start_ydstogo = config.get('start_ydstogo', None)
        self.start_yardline = config.get('start_yardline', None)
        
        # State dimensions - use consistent 11 dims for both players
        # This is needed for DMC compatibility (same shape for all players)
        # Offense: [down, ydstogo, yardline, formation_one_hot, box_count] = 11 dims
        # Defense: [down, ydstogo, yardline, formation_one_hot, 0] = padded to 11 dims
        self.state_shape = [[11], [11]]  # Same dims for both players (DMC compatible)
        self.action_shape = [None, None]
        
        # Encoding mappings
        self.formations = ("SHOTGUN", "SINGLEBACK", "I_FORM", "PISTOL", "EMPTY", "JUMBO", "WILDCAT")
        self.formation_to_idx = {f: i for i, f in enumerate(self.formations)}
    
    def reset(self):
        """Reset the environment and apply custom starting position if configured."""
        state, player_id = super().reset()
        
        # Apply custom starting position if configured
        if self.start_down is not None:
            self.game.down = self.start_down
        if self.start_ydstogo is not None:
            self.game.ydstogo = self.start_ydstogo
        if self.start_yardline is not None:
            self.game.yardline = self.start_yardline
        
        # Re-extract state if we modified game state
        if self.start_down or self.start_ydstogo or self.start_yardline:
            state = self._extract_state(self.game.get_state(player_id))
        
        return state, player_id
    
    def _extract_state(self, state):
        """Extract state features for neural network."""
        player_id = state.get('player_id', 0)
        phase = state.get('phase', 'formation')
        
        # Base features (normalized)
        down = state['down'] / 4.0
        ydstogo = min(state['ydstogo'], 30) / 30.0
        yardline = state['yardline'] / 100.0
        
        if phase == 'formation':
            # Phase 0: Offense picks formation, just sees game state
            obs = np.zeros(11, dtype=np.float32)
            obs[:3] = [down, ydstogo, yardline]
            legal_actions = {i: None for i in state['legal_actions']}
            
        elif phase == 'defense':
            # Phase 1: Defense sees formation (pad to 11 dims for consistency)
            formation = state.get('formation', 'SHOTGUN')
            formation_vec = self._encode_formation(formation)
            obs = np.zeros(11, dtype=np.float32)
            obs[:3] = [down, ydstogo, yardline]
            obs[3:10] = formation_vec
            # obs[10] left as 0 (no box info for defense)
            legal_actions = {i: None for i in state['legal_actions']}
            
        else:  # phase == 'play_type'
            # Phase 2: Offense sees box count + own formation
            formation = state.get('formation', 'SHOTGUN')
            box_count = state.get('box_count', 6)
            formation_vec = self._encode_formation(formation)
            box_normalized = (box_count - 4) / 4.0  # Normalize 4-8 to 0-1
            obs = np.array([down, ydstogo, yardline] + formation_vec + [box_normalized], dtype=np.float32)
            legal_actions = {i: None for i in state['legal_actions']}
        
        return {
            'obs': obs,
            'legal_actions': legal_actions,
            'raw_obs': state,
            'raw_legal_actions': state['legal_actions']
        }
    
    def _encode_formation(self, formation):
        """One-hot encode formation."""
        one_hot = [0] * len(self.formations)
        if formation in self.formation_to_idx:
            one_hot[self.formation_to_idx[formation]] = 1
        return one_hot
    
    def _decode_action(self, action_id):
        """Decode action ID to game action."""
        return action_id
    
    def _get_legal_actions(self):
        """Get legal actions for current player."""
        return self.game.get_legal_actions()
    
    def get_payoffs(self):
        """Get payoffs for all players."""
        return self.game.get_payoffs()
    
    def get_perfect_information(self):
        """Get perfect information state."""
        return {
            'down': self.game.down,
            'ydstogo': self.game.ydstogo,
            'yardline': self.game.yardline,
            'phase': self.game.phase,
            'pending_formation': self.game.pending_formation,
            'pending_defense_action': self.game.pending_defense_action,
            'current_player': self.game.current_player,
            'is_over': self.game.is_over()
        }
