"""
NFL IIG (Imperfect Information Game) - 3-Phase Version (Refactored: Scrimmage Only)

In this version, the offense commits to formation AND play type BEFORE
the defense chooses. Defense only sees the formation.

Refactor Update:
- Special Teams (Punt/FG) removed.
- 4th down failure = Turnover.
- Inherits config/rewards from NFLGame.

Phase 0: Offense → Formation
Phase 1: Offense → Play Type (pass/rush) - hidden from defense
Phase 2: Defense → Box count (sees formation only)
"""

import numpy as np
from rlcard.games.nfl.game import (
    NFLGame, 
    FORMATION_ACTIONS, 
    DEFENSE_ACTIONS, 
    PLAY_TYPE_ACTIONS,
    INITIAL_ACTIONS,
)


class NFLGameIIG(NFLGame):
    """NFL Game with imperfect information (huddle commit)."""
    
    def __init__(self, allow_step_back=False, data_path=None, 
                 single_play=False, start_down=1, use_cached_model=False, 
                 seed=None, reward_type='epa'):
        """Initialize IIG NFL game."""
        super().__init__(
            allow_step_back=allow_step_back,
            data_path=data_path,
            single_play=single_play,
            start_down=start_down,
            use_cached_model=use_cached_model,
            reward_type=reward_type
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
        """Return max action count (5 for phase 0/2, 2 for phase 1)."""
        return max(len(INITIAL_ACTIONS), len(DEFENSE_ACTIONS))
    
    def get_legal_actions(self):
        """Get legal actions based on phase."""
        if self.phase == 0:
            # Formation (5 actions)
            return list(range(len(INITIAL_ACTIONS)))
        elif self.phase == 1:
            # Play type selection (2 actions)
            return list(range(len(PLAY_TYPE_ACTIONS)))
        elif self.phase == 2:
            # Defense chooses box count (5 actions)
            return list(range(len(DEFENSE_ACTIONS)))
        else:
            return []
    
    def step(self, action):
        """Process IIG step."""
        if self.allow_step_back:
            self._save_state()
            
        if self.phase == 0: # Offense: Formation
            idx = action
            self.pending_formation = INITIAL_ACTIONS[idx]
            
            # Go to Phase 1 (Play Type selection, hidden)
            self.phase = 1
            self.current_player = 0 # Still offense
            
        elif self.phase == 1: # Offense: Play Type (Hidden commit)
            idx = action
            self.committed_play_type = PLAY_TYPE_ACTIONS[idx]
            
            # Go to Phase 2 (Defense box count)
            self.phase = 2
            self.current_player = 1 # Switch to defense
            
        elif self.phase == 2: # Defense: Box Count
            idx = action
            self.pending_defense_action = DEFENSE_ACTIONS[idx]
            
            # Execute committed play
            self._execute_committed_play()
            
        return self.get_state(self.current_player), self.current_player
    
    def get_state(self, player_id):
        """Get state for current player, handling IIG phases."""
        # This overrides the base class get_state to handle IIG-specific legal actions
        
        # Mapping phase to name
        phase_name = {
            0: 'formation',
            1: 'play_type',
            2: 'defense',
        }.get(self.phase, 'formation')

        obs = np.array([
            self.down,
            self.ydstogo,
            self.yardline,
            self.phase,
        ])

        legal_actions = list(self.get_legal_actions())
        # IIG specific action mapping
        if self.phase == 0:
            raw_legal_actions = [self.initial_actions[i] for i in legal_actions]
        elif self.phase == 1:
            raw_legal_actions = [self.play_type_actions[i] for i in legal_actions]
        else:
            raw_legal_actions = [self.defense_actions[i] for i in legal_actions]

        # Convert list to keys dict
        legal_actions_dict = {i: None for i in legal_actions}

        state = {
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': phase_name,
            'player_id': player_id,
            'legal_actions': legal_actions_dict,
            'obs': obs,
            'raw_obs': obs,
            'raw_legal_actions': raw_legal_actions,
        }
        return state
    
    def _execute_committed_play(self):
        """Execute the committed play against the defensive alignment."""
        
        # Build standard actions expected by base class
        # Offense: (Formation, PlayType)
        # Defense: (Box, Personnel)
        
        offense_action = (self.pending_formation, self.committed_play_type)
        defense_action = self.pending_defense_action
        
        result = self._get_outcome(
            self.down, self.ydstogo, self.yardline,
            offense_action, defense_action
        )
        
        yards_gained = int(round(result['yards_gained']))
        turnover = result.get('turnover', False)
        
        reward_offense = self._calculate_reward(yards_gained, turnover, is_touchdown=False)
        
        # Apply State Update
        if self.yardline + yards_gained >= 100: # TD
            if self.reward_type == 'touchdown': reward_offense = 1.0
            elif self.reward_type == 'yards': reward_offense = float(yards_gained)
            
            self.payoffs = [float(reward_offense), -float(reward_offense)]
            self.is_over_flag = True
            self.current_player = 0
            
        elif turnover:
            if self.reward_type == 'touchdown': reward_offense = 0.0
            self.payoffs = [float(reward_offense), -float(reward_offense)]
            self.is_over_flag = True
            
        else:
             self.yardline += yards_gained
             self.ydstogo -= yards_gained
             
             if self.ydstogo <= 0: # 1st Down
                 self.down = 1
                 dist_to_goal = 100 - self.yardline
                 self.ydstogo = min(10, dist_to_goal)
                 if self.single_play:
                     self.payoffs = [float(reward_offense), -float(reward_offense)]
                     self.is_over_flag = True
             else:
                 self.down += 1
                 if self.down > 4: # Turnover on Downs
                     if self.reward_type == 'epa':
                         ep_after = -self._calculate_ep(1, 10, 100 - self.yardline)
                         reward_offense = ep_after - self.ep_before
                         
                     self.payoffs = [float(reward_offense), -float(reward_offense)]
                     self.is_over_flag = True
        
        # Prepare for next play if not over
        if not self.is_over_flag:
            self.phase = 0
            self.current_player = 0
            self.committed_play_type = None
            self.pending_formation = None
            self.pending_defense_action = None
            self.ep_before = self._calculate_ep(self.down, self.ydstogo, self.yardline)

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
        self.committed_play_type = state['committed_play_type']
        return True
