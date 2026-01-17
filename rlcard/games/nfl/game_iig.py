"""
NFL IIG (Imperfect Information Game) - 3-Phase Version

In this version, the offense commits to formation AND play type BEFORE
the defense chooses. Defense only sees the formation.

New 3-Phase Structure (unified with standard game):
- Phase 0: Offense selects formation (or PUNT/FG)
- Phase 1: Offense selects play type (pass/rush) - defense doesn't see this
- Phase 2: Defense sees formation, selects box count

This keeps formation and play-type as separate decisions while both
are committed before defense acts. This unifies the action space
across all game variants.

Action spaces:
- Phase 0: 7 actions (5 formations + PUNT + FG)
- Phase 1: 2 actions (pass, rush)
- Phase 2: 5 actions (box counts 4-8)
"""

import numpy as np
from rlcard.games.nfl.game import (
    NFLGame, 
    FORMATION_ACTIONS, 
    DEFENSE_ACTIONS, 
    PLAY_TYPE_ACTIONS,
    SPECIAL_TEAMS,
    INITIAL_ACTIONS,
)


class NFLGameIIG(NFLGame):
    """NFL Game with imperfect information (huddle commit).
    
    3-phase game where offense commits both formation and play type
    before defense responds:
    
    Phase 0: Offense → Formation (or PUNT/FG)
    Phase 1: Offense → Play Type (pass/rush) - hidden from defense
    Phase 2: Defense → Box count (sees formation only)
    """
    
    def __init__(self, allow_step_back=False, data_path=None, 
                 single_play=False, start_down=1, use_cached_model=False, seed=None):
        """Initialize IIG NFL game."""
        super().__init__(
            allow_step_back=allow_step_back,
            data_path=data_path,
            single_play=single_play,
            start_down=start_down,
            use_cached_model=use_cached_model,
        )
        # Store committed play type (hidden from defense)
        self.committed_play_type = None
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
    
    def init_game(self):
        """Initialize a new game."""
        result = super().init_game()
        self.committed_play_type = None
        return result
    
    def get_num_actions(self):
        """Return max action count (7 for phase 0/2, 2 for phase 1)."""
        return max(len(INITIAL_ACTIONS), len(DEFENSE_ACTIONS))
    
    def get_legal_actions(self):
        """Get legal actions based on phase."""
        if self.phase == 0:
            # Formation + special teams (7 actions)
            return list(range(len(INITIAL_ACTIONS)))
        elif self.phase == 1:
            # Play type selection (2 actions)
            return list(range(len(PLAY_TYPE_ACTIONS)))
        elif self.phase == 2:
            # Defense chooses box count (5 actions)
            return list(range(len(DEFENSE_ACTIONS)))
        else:
            return []
    
    def _save_state(self):
        """Save current state for step_back."""
        self.history.append({
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'current_player': self.current_player,
            'phase': self.phase,
            'pending_formation': self.pending_formation,
            'pending_defense_action': self.pending_defense_action,
            'is_over_flag': self.is_over_flag,
            'payoffs': self.payoffs.copy(),
            'committed_play_type': self.committed_play_type,
        })
    
    def step_back(self):
        """Restore previous state."""
        if not self.history:
            return False
        
        state = self.history.pop()
        self.down = state['down']
        self.ydstogo = state['ydstogo']
        self.yardline = state['yardline']
        self.current_player = state['current_player']
        self.phase = state['phase']
        self.pending_formation = state['pending_formation']
        self.pending_defense_action = state['pending_defense_action']
        self.is_over_flag = state['is_over_flag']
        self.payoffs = state['payoffs']
        self.committed_play_type = state.get('committed_play_type')
        return True
    
    def step(self, action):
        """Process an action.
        
        Phase 0: Offense picks formation (or PUNT/FG)
        Phase 1: Offense picks play type (pass/rush) - hidden from defense
        Phase 2: Defense picks box count, then play executes
        """
        if self.allow_step_back:
            self._save_state()
        
        if self.phase == 0:
            # Offense phase 0 - pick formation or special teams
            action_str = action if isinstance(action, str) else INITIAL_ACTIONS[action]
            
            if action_str in SPECIAL_TEAMS:
                # Special teams - resolve immediately
                self._resolve_special_teams(action_str)
                return self.get_state(self.current_player), self.current_player
            
            # Store formation, move to play type selection
            self.pending_formation = action_str
            self.phase = 1
            # Still offense's turn (player 0)
            
        elif self.phase == 1:
            # Offense phase 1 - pick play type (hidden from defense)
            play_type = action if isinstance(action, str) else PLAY_TYPE_ACTIONS[action]
            self.committed_play_type = play_type
            
            # Move to defense phase
            self.phase = 2
            self.current_player = 1
            
        elif self.phase == 2:
            # Defense phase - picks box count
            self.pending_defense_action = DEFENSE_ACTIONS[action]
            
            # Execute the committed play
            self._execute_committed_play()
            
        return self.get_state(self.current_player), self.current_player
    
    def _execute_committed_play(self):
        """Execute the committed play against the defensive alignment."""
        old_ep = self._calculate_ep(
            self.down, self.ydstogo, self.yardline,
            goal_to_go=(100 - self.yardline) < self.ydstogo
        )
        
        play_type = self.committed_play_type
        formation = self.pending_formation
        defense_action = self.pending_defense_action
        
        result = self._get_outcome(
            self.down, self.ydstogo, self.yardline,
            formation, defense_action, play_type
        )
        
        yards = result['yards_gained']
        turnover = result.get('turnover', False)
        
        self._apply_outcome(yards, turnover, old_ep)
        
        # Reset for next play
        self.phase = 0
        self.current_player = 0
        self.pending_formation = None
        self.pending_defense_action = None
        self.committed_play_type = None
    
    def _get_outcome(self, down, ydstogo, yardline, formation, defense_action, play_type):
        """Get outcome - uses play_type directly."""
        if isinstance(defense_action, tuple):
            box_count = defense_action[0]
        elif isinstance(defense_action, int):
            box_count = defense_action
        else:
            box_count = 6
        
        if self.cached_model is not None:
            return self.cached_model.sample(
                formation, play_type, box_count, yardline, down, ydstogo
            )
        
        if self.outcome_model is not None:
            return self.outcome_model.sample(
                formation, play_type, box_count, yardline, down, ydstogo
            )
        
        offense_action = (formation, play_type)
        return super()._get_outcome(down, ydstogo, yardline, offense_action, defense_action)
    
    def _apply_outcome(self, yards, turnover, old_ep):
        """Apply outcome and calculate EPA."""
        # Ensure yards is int
        yards = int(np.round(yards))
        
        if turnover:
            self.is_over_flag = True
            opp_yardline = 100 - (self.yardline + yards)
            opp_yardline = int(np.round(opp_yardline))
            opp_yardline = max(1, min(99, opp_yardline))
            opp_ep = self._calculate_ep(1, 10, opp_yardline)
            epa = -opp_ep - old_ep
        elif self.yardline + yards >= 100:
            self.is_over_flag = True
            epa = 7.0 - old_ep
        elif yards >= self.ydstogo:
            self.yardline += yards
            self.down = 1
            self.ydstogo = min(10, 100 - self.yardline)
            new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            epa = new_ep - old_ep
        else:
            self.yardline += yards
            self.down += 1
            self.ydstogo -= yards
            
            if self.down > 4:
                self.is_over_flag = True
                opp_yardline = max(1, min(99, 100 - self.yardline))
                opp_ep = self._calculate_ep(1, 10, opp_yardline)
                epa = -opp_ep - old_ep
            else:
                new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
                epa = new_ep - old_ep
        
        if self.yardline <= 0:
            self.is_over_flag = True
            opp_ep = self._calculate_ep(1, 10, 35)
            epa = -(2.0 + opp_ep) - old_ep
        
        if self.single_play:
            self.is_over_flag = True
        
        self.payoffs = [epa, -epa]
    
    def get_state(self, player_id):
        """Get state from perspective of player.
        
        Key: Defense (player 1, phase 2) sees formation but NOT play type.
        """
        legal_actions = self.get_legal_actions()
        
        # Build 12-dim observation array
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = self.down / 4.0
        obs[1] = min(self.ydstogo, 30) / 30.0
        obs[2] = self.yardline / 100.0
        
        # Formation encoding (indices 3-7) - visible after phase 0
        if self.phase >= 1 and self.pending_formation in FORMATION_ACTIONS:
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation)
            obs[3 + formation_idx] = 1.0
        
        # Phase encoding (index 11)
        obs[11] = self.phase / 2.0
        
        # Get raw legal action names and phase name
        if self.phase == 0:
            raw_legal_actions = list(INITIAL_ACTIONS)
            phase_name = 'formation'
        elif self.phase == 1:
            raw_legal_actions = list(PLAY_TYPE_ACTIONS)
            phase_name = 'play_type'
        elif self.phase == 2:
            raw_legal_actions = [f"{d[0]}_box" for d in DEFENSE_ACTIONS]
            phase_name = 'defense'
        else:
            raw_legal_actions = []
            phase_name = 'unknown'
        
        # Convert legal_actions to dict
        legal_actions_dict = {i: None for i in legal_actions}
        
        state = {
            'obs': obs,
            'legal_actions': legal_actions_dict,
            'raw_legal_actions': raw_legal_actions,
            'player_id': player_id,
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': self.phase,
            'phase_name': phase_name,
        }
        
        # Add formation if visible (phases 1-2)
        if self.phase >= 1:
            state['formation'] = self.pending_formation
        
        return state


# Export action mappings for analysis
def decode_iig_action(phase, action_idx):
    """Decode action index based on phase."""
    if phase == 0:
        return INITIAL_ACTIONS[action_idx]
    elif phase == 1:
        return PLAY_TYPE_ACTIONS[action_idx]
    elif phase == 2:
        return DEFENSE_ACTIONS[action_idx]
    return None


# Legacy exports for backwards compatibility
IIG_ACTION_NAMES = list(INITIAL_ACTIONS) + list(PLAY_TYPE_ACTIONS)
