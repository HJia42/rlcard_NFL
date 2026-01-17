"""
NFL IIG Scrimmage - Pure Formation vs Box Strategy

This simplified IIG game focuses on the core strategic interaction
with scrimmage plays only (no special teams):
- Phase 0: Offense picks formation (5 actions)
- Phase 1: Offense picks play type (pass/rush) - hidden from defense
- Phase 2: Defense sees formation, picks box count

This creates TRUE imperfect information since:
- Offense commits without knowing defense
- Defense observes formation but NOT the hidden play type

Note: For games WITH special teams, use NFLGameIIG instead.
"""

import numpy as np
from rlcard.games.nfl.game import (
    NFLGame, 
    FORMATION_ACTIONS, 
    DEFENSE_ACTIONS, 
    PLAY_TYPE_ACTIONS,
)


class NFLGameIIGScrimmage(NFLGame):
    """NFL IIG game without special teams - pure scrimmage plays.
    
    3-phase game where offense commits both formation and play type
    before defense responds:
    
    Phase 0: Offense → Formation (5 actions, no PUNT/FG)
    Phase 1: Offense → Play Type (pass/rush) - hidden from defense
    Phase 2: Defense → Box count (sees formation only)
    """
    
    def __init__(self, allow_step_back=False, data_path=None, 
                 single_play=True, use_cached_model=False, seed=None):
        """Initialize scrimmage-only IIG game."""
        super().__init__(
            allow_step_back=allow_step_back,
            data_path=data_path,
            single_play=single_play,
            use_cached_model=use_cached_model,
        )
        self.committed_play_type = None
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
    
    def init_game(self):
        """Initialize a new game."""
        result = super().init_game()
        self.committed_play_type = None
        return result
    
    def get_num_actions(self):
        """Return max action count (5 formations or 5 box counts)."""
        return max(len(FORMATION_ACTIONS), len(DEFENSE_ACTIONS))
    
    def get_legal_actions(self):
        """Get legal actions based on phase."""
        if self.phase == 0:
            # Formations only (no special teams)
            return list(range(len(FORMATION_ACTIONS)))
        elif self.phase == 1:
            # Play type selection
            return list(range(len(PLAY_TYPE_ACTIONS)))
        elif self.phase == 2:
            # Defense box count
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
        
        Phase 0: Offense picks formation (no special teams)
        Phase 1: Offense picks play type - hidden from defense  
        Phase 2: Defense picks box count, then play executes
        """
        if self.allow_step_back:
            self._save_state()
        
        if self.phase == 0:
            # Formation selection
            formation = action if isinstance(action, str) else FORMATION_ACTIONS[action]
            self.pending_formation = formation
            self.phase = 1
            # Still offense's turn
            
        elif self.phase == 1:
            # Play type selection (hidden from defense)
            play_type = action if isinstance(action, str) else PLAY_TYPE_ACTIONS[action]
            self.committed_play_type = play_type
            self.phase = 2
            self.current_player = 1
            
        elif self.phase == 2:
            # Defense picks box count
            self.pending_defense_action = DEFENSE_ACTIONS[action]
            self._execute_committed_play()
            
        return self.get_state(self.current_player), self.current_player
    
    def _execute_committed_play(self):
        """Execute the committed play against the defensive alignment."""
        old_ep = self._calculate_ep(
            self.down, self.ydstogo, self.yardline,
            goal_to_go=(100 - self.yardline) < self.ydstogo
        )
        
        result = self._get_outcome(
            self.down, self.ydstogo, self.yardline,
            self.pending_formation, self.pending_defense_action, 
            self.committed_play_type
        )
        
        yards = result['yards_gained']
        turnover = result.get('turnover', False)
        
        self._apply_outcome(yards, turnover, old_ep)
        
        self.phase = 0
        self.current_player = 0
        self.pending_formation = None
        self.pending_defense_action = None
        self.committed_play_type = None
    
    def _get_outcome(self, down, ydstogo, yardline, formation, defense_action, play_type):
        """Get outcome using play_type directly."""
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
        if turnover:
            self.is_over_flag = True
            opp_yardline = max(1, min(99, 100 - (self.yardline + yards)))
            opp_ep = self._calculate_ep(1, 10, opp_yardline)
            epa = -opp_ep - old_ep
        elif self.yardline + yards >= 100:
            self.is_over_flag = True
            epa = 7.0 - old_ep
        elif yards >= self.ydstogo:
            self.yardline += yards
            self.down = 1
            self.ydstogo = min(10, 100 - self.yardline)
            epa = self._calculate_ep(self.down, self.ydstogo, self.yardline) - old_ep
        else:
            self.yardline += yards
            self.down += 1
            self.ydstogo -= yards
            
            if self.down > 4:
                self.is_over_flag = True
                opp_yardline = max(1, min(99, 100 - self.yardline))
                epa = -self._calculate_ep(1, 10, opp_yardline) - old_ep
            else:
                epa = self._calculate_ep(self.down, self.ydstogo, self.yardline) - old_ep
        
        if self.yardline <= 0:
            self.is_over_flag = True
            epa = -(2.0 + self._calculate_ep(1, 10, 35)) - old_ep
        
        if self.single_play:
            self.is_over_flag = True
        
        self.payoffs = [epa, -epa]
    
    def get_state(self, player_id):
        """Get state from perspective of player."""
        legal_actions = self.get_legal_actions()
        
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = self.down / 4.0
        obs[1] = min(self.ydstogo, 30) / 30.0
        obs[2] = self.yardline / 100.0
        
        if self.phase >= 1 and self.pending_formation in FORMATION_ACTIONS:
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation)
            obs[3 + formation_idx] = 1.0
        
        obs[11] = self.phase / 2.0
        
        if self.phase == 0:
            raw_legal_actions = list(FORMATION_ACTIONS)
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
        
        if self.phase >= 1:
            state['formation'] = self.pending_formation
        
        return state


# Action names for analysis
SCRIMMAGE_ACTION_NAMES = list(FORMATION_ACTIONS) + list(PLAY_TYPE_ACTIONS)
