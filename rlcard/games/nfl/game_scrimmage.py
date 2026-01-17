
import numpy as np
from rlcard.games.nfl.game_iig_scrimmage import NFLGameIIGScrimmage

class NFLGameScrimmage(NFLGameIIGScrimmage):
    """
    NFL Perfect Information Scrimmage Game.
    
    Identical to NFLGameIIGScrimmage (No Punt/FG), but turn order is reorganized
    to provide PERFECT INFORMATION:
    
    1. Phase 0: Offense chooses FORMATION.
    2. Phase 1: Defense sees FORMATION, chooses BOX COUNT.
    3. Phase 2: Offense sees BOX COUNT, chooses PLAY TYPE (Pass/Rush).
    """

    def get_legal_actions(self):
        """Get legal actions based on phase."""
        if self.phase == 0:
            # Offense: Formations only
            return list(range(self.num_formation_actions))
        elif self.phase == 1:
            # Defense: Box count
            return list(range(self.num_defense_actions))
        elif self.phase == 2:
            # Offense: Play type
            return list(range(self.num_play_type_actions))
        else:
            return []

    def step(self, action):
        """Process action with Perfect Information sequence."""
        if self.allow_step_back:
            self._save_state()
            
        if self.phase == 0:
            # Phase 0: Offense picks formation -> Defense's turn
            self.pending_formation = self.formation_actions[action]
            self.phase = 1
            self.current_player = 1 # Defense turn
            
        elif self.phase == 1:
            # Phase 1: Defense picks box count (knowing formation) -> Offense's turn
            self.pending_defense_action = self.defense_actions[action]
            self.phase = 2
            self.current_player = 0 # Offense turn
            
        elif self.phase == 2:
            # Phase 2: Offense picks play type (knowing box count) -> Resolve
            play_type = self.play_type_actions[action]
            
            # Resolve outcome
            offense_action = (self.pending_formation, play_type)
            defense_action = self.pending_defense_action
            
            outcome = self._get_outcome(
                self.down, self.ydstogo, self.yardline,
                offense_action, defense_action
            )
            
            # Apply outcome (using integer rounding logic from parent/grandparent)
            self._apply_outcome(outcome['yards_gained'], outcome['turnover'], self.ep_before)
            
            self.is_over_flag = True
            
        return self.get_state(self.current_player), self.current_player
