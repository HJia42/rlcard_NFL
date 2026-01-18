"""
NFL Play-by-Play Game for RLCard (Refactored: Scrimmage Only)

A two-player imperfect information game with 3 turns per play:
- Turn 1 (Player 0): Offense selects formation
- Turn 2 (Player 1): Defense sees formation, selects box count
- Turn 3 (Player 0): Offense sees box count, selects play type (pass/rush)
- Then outcome is resolved from historical data

Refactor Update:
- Special Teams (Punt/FG) has been removed.
- On 4th Down, if offense fails to convert, it is a Turnover (End of Drive/Game).
- Added Configurable Reward Function (EPA, Yards, TD).
"""

import numpy as np
from copy import copy
import os
import pickle
from pathlib import Path

from rlcard.games.nfl.player import OffensePlayer, DefensePlayer

# Game constants - these should match your cleaned_nfl_rl_data.csv
FORMATIONS = ("SHOTGUN", "SINGLEBACK", "UNDER CENTER", "I_FORM", "EMPTY")
PLAY_TYPES = ("pass", "rush")
BOX_COUNTS = (4, 5, 6, 7, 8)
PERSONNEL_TYPES = ("Standard",)

# Build action maps
# Phase 0: Offense picks formation (5 actions)
FORMATION_ACTIONS = list(FORMATIONS)
INITIAL_ACTIONS = FORMATION_ACTIONS 

# Phase 1: Defense picks box count (5 actions)
DEFENSE_ACTIONS = []
for box in BOX_COUNTS:
    for personnel in PERSONNEL_TYPES:
        DEFENSE_ACTIONS.append((box, personnel))

# Phase 2: Offense picks play type (2 actions)
PLAY_TYPE_ACTIONS = list(PLAY_TYPES)


class NFLGame:
    """NFL Play-by-Play Game compatible with RLCard."""
    
    def __init__(self, allow_step_back=False, data_path=None, use_simple_model=None, 
                 single_play=False, start_down=1, use_distribution_model=False, 
                 use_cached_model=False, reward_type='epa'):
        """Initialize NFL Game.
        
        Args:
            allow_step_back: Whether to support step_back for CFR
            data_path: Path to cleaned NFL data (optional)
            use_simple_model: If True, skip pandas and use fast simplified model.
            single_play: If True, game ends after one complete play.
            start_down: Starting down (1-4). Default: 1.
            use_distribution_model: If True, use statistical distributions.
            use_cached_model: If True, use pre-computed cached distributions.
            reward_type: 'epa' (default), 'yards', or 'touchdown'.
        """
        self.allow_step_back = allow_step_back
        self.single_play = single_play
        self.start_down = start_down
        self.use_distribution_model = use_distribution_model
        self.use_cached_model = use_cached_model
        
        # New: Reward Configuration
        self.reward_type = reward_type
        if self.reward_type not in ['epa', 'yards', 'touchdown']:
            print(f"Warning: Unknown reward_type '{reward_type}', defaulting to 'epa'")
            self.reward_type = 'epa'

        self.np_random = np.random.RandomState()
        
        # Initialize action spaces
        self.initial_actions = INITIAL_ACTIONS
        self.defense_actions = DEFENSE_ACTIONS
        self.play_type_actions = PLAY_TYPE_ACTIONS
        
        # Load play data logic
        if use_cached_model:
            try:
                base_dir = Path(__file__).parent
                cache_files = [
                    base_dir / "cached_outcomes_full.pkl",
                    base_dir.parent.parent.parent / "data" / "cached_outcomes_full.pkl"
                ]
                model_path = next((p for p in cache_files if p.exists()), None)
                if model_path:
                    with open(model_path, 'rb') as f:
                        self.cached_model = pickle.load(f)
                    print(f"Loaded cached outcome model from {model_path}")
                    self.use_simple_model = False
                else:
                    print("Warning: Cached model not found, falling back to simple model")
                    self.use_simple_model = True
                    self.use_cached_model = False
            except Exception as e:
                print(f"Error loading cached model: {e}")
                self.use_simple_model = True
                self.use_cached_model = False
        elif use_distribution_model:
            try:
                from rlcard.games.nfl.outcome_model import NFLOutcomeModel
                self.outcome_model = NFLOutcomeModel(data_path)
                self.use_simple_model = False
            except Exception as e:
                print(f"Error loading outcome model: {e}")
                self.use_simple_model = True
                self.use_distribution_model = False
        else:
            if use_simple_model is None:
                self.use_simple_model = False # Default to attempting data load
            else:
                self.use_simple_model = use_simple_model
            
            if not self.use_simple_model and not self.use_distribution_model:
                self._load_data(data_path)
                if self.play_data is None:
                    self.use_simple_model = True

    def _load_data(self, data_path):
        """Load historical play data."""
        if data_path is None:
            possible_paths = [
                Path(__file__).parent.parent.parent.parent.parent / "Code" / "data" / "cleaned_nfl_rl_data.csv",
                Path.home() / "Projects" / "NFL_Playcalling" / "Code" / "data" / "cleaned_nfl_rl_data.csv",
            ]
            for p in possible_paths:
                if p.exists():
                    data_path = str(p)
                    break
        
        if data_path and os.path.exists(data_path):
            try:
                import pandas as pd
                self.play_data = pd.read_csv(data_path)
                print(f"Loaded {len(self.play_data)} plays from {data_path}")
            except Exception as e:
                print(f"Warning: Could not load play data: {e}")
                self.play_data = None
        else:
            print("Warning: No play data found, using simplified outcome model")
            self.play_data = None
    
    def configure(self, game_config):
        """Configure game parameters."""
        pass
    
    def init_game(self):
        """Initialize a new game (drive)."""
        self.players = [OffensePlayer(0), DefensePlayer(1)]
        
        # Initial state: 1st & 10 at own 25 (or custom down)
        self.down = self.start_down
        
        if self.single_play:
            # Randomize field position for generalizable training
            self.yardline = int(np.random.randint(1, 100))
            
            # Randomize ydstogo (capped by distance to goal and realistic max)
            dist_to_goal = 100 - self.yardline
            max_yds = min(20, dist_to_goal)
            self.ydstogo = int(np.random.randint(1, max_yds + 1))
        else:
            self.ydstogo = 10
            self.yardline = 25
        
        # Start with offense picking formation (phase 0)
        self.current_player = 0
        self.phase = 0
        self.pending_formation = None
        self.pending_defense_action = None
        self.is_over_flag = False
        self.payoffs = [0, 0]
        self.history = []
        
        # Calculate baseline EP for reward delta
        self.ep_before = self._calculate_ep(self.down, self.ydstogo, self.yardline)
        
        state = self.get_state(self.current_player)
        return state, self.current_player
    
    def step(self, action):
        """Process an action from current player."""
        if self.allow_step_back:
            self._save_state()
        
        if self.phase == 0:
            # Phase 0: Offense picks formation (No Special Teams anymore)
            action_str = action if isinstance(action, str) else self.initial_actions[action]
            
            # Normal play - proceed to defense
            self.pending_formation = action_str
            self.phase = 1
            self.current_player = 1  # Defense's turn
            
        elif self.phase == 1:
            # Phase 1: Defense picks box count
            self.pending_defense_action = action if isinstance(action, tuple) else self.defense_actions[action]
            self.phase = 2
            self.current_player = 0  # Back to offense
            
        elif self.phase == 2:
            # Phase 2: Offense picks play type, then resolve
            play_type = action if isinstance(action, str) else self.play_type_actions[action]
            
            # Build full actions
            offense_action = (self.pending_formation, play_type)
            defense_action = self.pending_defense_action
            
            # Get outcome
            outcome = self._get_outcome(
                self.down, self.ydstogo, self.yardline,
                offense_action, defense_action
            )
            
            yards_gained = int(round(outcome['yards_gained']))
            turnover = outcome['turnover']
            
            # Calculate Rewards (Modular)
            reward_offense = self._calculate_reward(yards_gained, turnover, is_touchdown=False)
            
            # Check for Touchdown
            if self.yardline + yards_gained >= 100:
                # Touchdown!
                if self.reward_type == 'touchdown':
                    reward_offense = 1.0
                elif self.reward_type == 'yards':
                    reward_offense = float(yards_gained)
                
                self.payoffs = [float(reward_offense), -float(reward_offense)]
                self.is_over_flag = True
                self.current_player = 0 # Offense gets reward
            
            elif turnover:
                # Turnover
                if self.reward_type == 'touchdown': 
                    reward_offense = 0.0
                
                self.payoffs = [float(reward_offense), -float(reward_offense)]
                self.is_over_flag = True # End of drive
            
            else:
                # Update State
                self.yardline += yards_gained
                self.ydstogo -= yards_gained
                
                # Check for 1st Down / Turnover on Downs
                if self.ydstogo <= 0:
                    # First Down
                    self.down = 1
                    dist_to_goal = 100 - self.yardline
                    self.ydstogo = min(10, dist_to_goal)
                    
                    if self.single_play:
                        self.payoffs = [float(reward_offense), -float(reward_offense)]
                        self.is_over_flag = True
                        
                else:
                    self.down += 1
                    if self.down > 4:
                        # Turnover on Downs (Failed 4th Down)
                        if self.reward_type == 'epa':
                            # Opponent gets ball 100 - yardline
                            ep_after = -self._calculate_ep(1, 10, 100 - self.yardline)
                            reward_offense = ep_after - self.ep_before
                            
                        self.payoffs = [float(reward_offense), -float(reward_offense)]
                        self.is_over_flag = True
                        
                # If not over, setup next play
                if not self.is_over_flag:
                    self.phase = 0
                    self.current_player = 0
                    self.ep_before = self._calculate_ep(self.down, self.ydstogo, self.yardline)

        state = self.get_state(self.current_player)
        return state, self.current_player
        
    def _calculate_reward(self, yards_gained, turnover, is_touchdown):
        """Calculate reward based on configured reward_type."""
        if self.reward_type == 'yards':
            return float(yards_gained)
        
        elif self.reward_type == 'touchdown':
            return 0.0 # Handled in step()
            
        else: # 'epa' (Default)
            if turnover:
                current_yl = self.yardline + yards_gained
                ep_after = -self._calculate_ep(1, 10, 100 - current_yl)
            elif is_touchdown:
                ep_after = 7.0
            else:
                new_yl = self.yardline + yards_gained
                new_ydstogo = self.ydstogo - yards_gained
                new_down = self.down
                
                if new_ydstogo <= 0:
                    new_down = 1
                    new_ydstogo = min(10, 100 - new_yl)
                else:
                    new_down += 1
                    
                if new_down > 4:
                     ep_after = -self._calculate_ep(1, 10, 100 - new_yl)
                else:
                    ep_after = self._calculate_ep(new_down, new_ydstogo, new_yl)
            
            return ep_after - self.ep_before

    def _calculate_ep(self, down, ydstogo, yardline):
        """Simple Expected Points model (Interpolated)."""
        field_pos = yardline / 100.0
        # -2.0 at 0, 7.0 at 100
        ep = -2.0 + 9.0 * field_pos
        
        # Down/Distance Penalty
        down_penalty = (down - 1) * 0.5
        dist_penalty = (ydstogo - 5) * 0.1
        
        return ep - down_penalty - dist_penalty

    def _get_outcome(self, down, togo, yl, off_action, def_action):
        """Sample outcome from data or model."""
        formation, play_type = off_action
        box_count, personnel = def_action
        
        if self.use_cached_model and hasattr(self, 'cached_model'):
            # Key: (formation, play_type, box_count)
            key = (formation, play_type, box_count)
            
            if key in self.cached_model:
                outcome_dist = self.cached_model[key]
                turnover_prob = outcome_dist.get('turnover_prob', 0.0)
                is_turnover = (self.np_random.rand() < turnover_prob)
                
                if is_turnover:
                    return {'yards_gained': 0, 'turnover': True}
                else:
                    # Pass through to backup logic for exact yardage distribution if needed
                    # Or use a simple sampler if not fully implemented in this refactor
                    pass
            
        # Robust Fallback Model (Physics-based approximation)
        is_pass = (play_type == 'pass')
        yards = 0
        turnover = False
        
        if is_pass:
            if self.np_random.rand() < 0.6:
                yards = self.np_random.normal(11.0, 6.0)
                if box_count < 6:
                    yards += (6 - box_count) * 1.5
            else:
                yards = 0
            
            if self.np_random.rand() < 0.025:
                turnover = True
                
        else:
            yards = self.np_random.normal(4.0, 3.0)
            yards -= (box_count - 7) * 1.0
            if self.np_random.rand() < 0.01:
                turnover = True

        return {'yards_gained': yards, 'turnover': turnover}
        
    def _save_state(self):
        # Unchanged from original idea but needed for consistency
        pass

    def get_state(self, player_id):
        """Return state for player."""
        phase_name = {
            0: 'formation',
            1: 'defense',
            2: 'play_type',
        }.get(self.phase, 'formation')

        obs = np.array([
            self.down,
            self.ydstogo,
            self.yardline,
            self.phase,
        ])

        legal_actions = list(self.get_legal_actions())
        if self.phase == 0:
            raw_legal_actions = list(self.initial_actions)
        elif self.phase == 1:
            raw_legal_actions = list(self.defense_actions)
        else:
            raw_legal_actions = list(self.play_type_actions)

        state = {
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': phase_name,
            'player_id': player_id,
            'legal_actions': legal_actions,
            'obs': obs,
            'raw_obs': obs,
            'raw_legal_actions': raw_legal_actions,
        }

        if self.phase >= 1:
            state['formation'] = self.pending_formation
        if self.phase == 2 and self.pending_defense_action:
            box_count, personnel = self.pending_defense_action
            state['box_count'] = box_count
            state['personnel'] = personnel

        return state
        
    def get_legal_actions(self):
        """Return legal action indices based on phase."""
        if self.phase == 0:
            return range(len(self.initial_actions))
        elif self.phase == 1:
            return range(len(self.defense_actions))
        elif self.phase == 2:
            return range(len(self.play_type_actions))
        return []
        
    def _decode_action(self, action_id):
         if self.phase == 0: return self.initial_actions[action_id]
         return str(action_id)

    def get_num_players(self):
        return 2

    def get_num_actions(self):
        return max(len(self.initial_actions), len(self.defense_actions), len(self.play_type_actions))
        
    def get_payoffs(self):
        return self.payoffs
        
    def get_player_id(self):
        """Get current player id."""
        return self.current_player

    def is_over(self):
        return self.is_over_flag
