"""
NFL IIG (Imperfect Information Game) - Huddle Commit Version

In this version, the offense commits to BOTH formation AND play type
before the defense chooses. The defense only sees the formation.

This models "calling the play in the huddle" vs "audibling at the line".

Key differences from standard NFL game:
- Phase 0: Offense chooses (Formation × PlayType) or special teams
- Phase 1: Defense sees Formation only (NOT play type), chooses box count
- Phase 2: No decision - play executes automatically

Action space:
- Offense: 12 actions (5 formations × 2 play types + PUNT + FG)
- Defense: 5 actions (box counts 4-8)

This creates TRUE imperfect information since:
- Offense commits without knowing defense
- Defense observes only the "signal" (formation), not the hidden action (play type)
"""

import numpy as np
from rlcard.games.nfl.game import (
    NFLGame, 
    FORMATION_ACTIONS, 
    DEFENSE_ACTIONS, 
    PLAY_TYPE_ACTIONS,
    SPECIAL_TEAMS,
)

# IIG action space: Formation × PlayType + Special Teams
# 5 formations × 2 play types = 10
# + PUNT + FG = 12 total
IIG_FORMATION_ACTIONS = FORMATION_ACTIONS  # 5 formations
IIG_PLAY_TYPES = PLAY_TYPE_ACTIONS  # ['pass', 'rush']
IIG_SPECIAL_TEAMS = SPECIAL_TEAMS  # ['PUNT', 'FG']

# Combined actions for offense
IIG_OFFENSE_ACTIONS = []
for formation in IIG_FORMATION_ACTIONS:
    for play_type in IIG_PLAY_TYPES:
        IIG_OFFENSE_ACTIONS.append((formation, play_type))
# Add special teams
IIG_OFFENSE_ACTIONS.append(('PUNT', None))
IIG_OFFENSE_ACTIONS.append(('FG', None))

# Action name mapping for readability
IIG_ACTION_NAMES = [
    f"{f}_{pt}" if pt else f for f, pt in IIG_OFFENSE_ACTIONS
]


class NFLGameIIG(NFLGame):
    """NFL Game with imperfect information (huddle commit).
    
    The offense commits to both formation AND play type before
    seeing the defense's response. This creates a signaling game.
    """
    
    def __init__(self, allow_step_back=False, data_path=None, 
                 single_play=False, use_cached_model=False, seed=None):
        """Initialize IIG NFL game."""
        super().__init__(
            allow_step_back=allow_step_back,
            data_path=data_path,
            single_play=single_play,
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
        """Return max action count (12 for offense IIG)."""
        return len(IIG_OFFENSE_ACTIONS)
    
    def get_legal_actions(self):
        """Get legal actions based on phase."""
        if self.phase == 0:
            # All 12 IIG actions available
            return list(range(len(IIG_OFFENSE_ACTIONS)))
        elif self.phase == 1:
            # Defense chooses box count
            return list(range(len(DEFENSE_ACTIONS)))
        else:
            # Phase 2: No decision (auto-execute)
            return []
    
    def step(self, action):
        """Process an action.
        
        Phase 0: Offense commits to (Formation, PlayType) or special teams
        Phase 1: Defense sees Formation, picks box count
        Phase 2: Play executes (no player action needed)
        """
        if self.phase == 0:
            # Offense phase - commit to formation AND play type
            formation, play_type = IIG_OFFENSE_ACTIONS[action]
            
            if formation in SPECIAL_TEAMS:
                # Special teams - resolve immediately
                self._resolve_special_teams(formation)
                return self.get_state(self.current_player), self.current_player
            
            # Store both formation and (hidden) play type
            self.pending_formation = formation
            self.committed_play_type = play_type
            
            # Move to defense phase
            self.phase = 1
            self.current_player = 1
            
        elif self.phase == 1:
            # Defense phase - picks box count
            if self.allow_step_back:
                self._save_state()
            
            self.pending_defense_action = DEFENSE_ACTIONS[action]
            
            # Move to execution phase (auto-resolve, no player action)
            self.phase = 2
            self.current_player = 0
            
            # Auto-execute the play since play type was already committed
            self._execute_committed_play()
            
        return self.get_state(self.current_player), self.current_player
    
    def _execute_committed_play(self):
        """Execute the committed play against the defensive alignment."""
        # Calculate EPA before play
        old_ep = self._calculate_ep(
            self.down, self.ydstogo, self.yardline,
            goal_to_go=(100 - self.yardline) < self.ydstogo
        )
        
        # Get outcome using the COMMITTED play type
        play_type = self.committed_play_type  # 'pass' or 'rush'
        formation = self.pending_formation
        defense_action = self.pending_defense_action
        
        result = self._get_outcome(
            self.down, self.ydstogo, self.yardline,
            formation, defense_action, play_type
        )
        
        yards = result['yards_gained']
        turnover = result.get('turnover', False)
        
        # Update game state
        self._apply_outcome(yards, turnover, old_ep)
        
        # Reset for next play
        self.phase = 0
        self.current_player = 0
        self.pending_formation = None
        self.pending_defense_action = None
        self.committed_play_type = None
    
    def _get_outcome(self, down, ydstogo, yardline, formation, defense_action, play_type):
        """Get outcome - override to accept play_type directly."""
        if self.outcome_model:
            box_count = defense_action[0] if isinstance(defense_action, tuple) else defense_action
            return self.outcome_model.sample(
                formation, play_type, box_count,
                yardline, down, ydstogo
            )
        return super()._get_outcome(down, ydstogo, yardline, formation, defense_action)
    
    def _apply_outcome(self, yards, turnover, old_ep):
        """Apply outcome and calculate EPA."""
        if turnover:
            self.is_over_flag = True
            opp_yardline = 100 - (self.yardline + yards)
            opp_yardline = max(1, min(99, opp_yardline))
            opp_ep = self._calculate_ep(1, 10, opp_yardline)
            epa = -opp_ep - old_ep
        elif self.yardline + yards >= 100:
            # Touchdown
            self.is_over_flag = True
            epa = 7.0 - old_ep
        elif yards >= self.ydstogo:
            # First down
            self.yardline += yards
            self.down = 1
            self.ydstogo = min(10, 100 - self.yardline)
            new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            epa = new_ep - old_ep
        else:
            # Failed to convert
            self.yardline += yards
            self.down += 1
            self.ydstogo -= yards
            
            if self.down > 4:
                # Turnover on downs
                self.is_over_flag = True
                opp_yardline = 100 - self.yardline
                opp_yardline = max(1, min(99, opp_yardline))
                opp_ep = self._calculate_ep(1, 10, opp_yardline)
                epa = -opp_ep - old_ep
            else:
                new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
                epa = new_ep - old_ep
        
        # Safety check
        if self.yardline <= 0:
            self.is_over_flag = True
            opp_ep = self._calculate_ep(1, 10, 35)
            epa = -(2.0 + opp_ep) - old_ep
        
        if self.single_play:
            self.is_over_flag = True
        
        self.payoffs = [epa, -epa]
    
    def get_state(self, player_id):
        """Get state from perspective of player.
        
        Key difference: Defense (player 1) does NOT see the committed play type.
        Returns state dict with 'obs' array for neural network compatibility.
        """
        legal_actions = self.get_legal_actions()
        
        # Build observation array (12 dimensions)
        obs = np.zeros(12, dtype=np.float32)
        obs[0] = self.down / 4.0  # Normalized down
        obs[1] = min(self.ydstogo, 30) / 30.0  # Normalized yards to go
        obs[2] = self.yardline / 100.0  # Normalized yardline
        
        # Formation encoding (indices 3-7) - only visible in phase 1
        if self.phase == 1 and self.pending_formation in FORMATION_ACTIONS:
            formation_idx = FORMATION_ACTIONS.index(self.pending_formation)
            obs[3 + formation_idx] = 1.0
        
        # Box count (index 10) - not used in IIG since no phase 2 decision
        # Phase encoding (index 11)
        obs[11] = self.phase / 2.0
        
        # Get raw legal action names
        if self.phase == 0:
            raw_legal_actions = IIG_ACTION_NAMES
        elif self.phase == 1:
            raw_legal_actions = [f"{d[0]}_box" for d in DEFENSE_ACTIONS]
        else:
            raw_legal_actions = []
        
        return {
            'obs': obs,
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': self.phase,
            'legal_actions': legal_actions,
            'raw_legal_actions': raw_legal_actions,
            'player_id': player_id,
            # Defense does NOT see committed_play_type
        }


# Export action mappings for analysis
def decode_iig_action(action_idx):
    """Decode IIG action index to (formation, play_type) tuple."""
    return IIG_OFFENSE_ACTIONS[action_idx]


def encode_iig_action(formation, play_type):
    """Encode (formation, play_type) to action index."""
    target = (formation, play_type) if play_type else (formation, None)
    return IIG_OFFENSE_ACTIONS.index(target)
