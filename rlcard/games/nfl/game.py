"""
NFL Play-by-Play Game for RLCard

A two-player imperfect information game with 3 turns per play:
- Turn 1 (Player 0): Offense selects formation
- Turn 2 (Player 1): Defense sees formation, selects box count
- Turn 3 (Player 0): Offense sees box count, selects play type (pass/rush)
- Then outcome is resolved from historical data

This models the NFL audible system where QB reads the defense.
"""

import numpy as np
from copy import copy
import os
import pickle
from pathlib import Path

from rlcard.games.nfl.player import OffensePlayer, DefensePlayer


# Game constants
FORMATIONS = ("SHOTGUN", "SINGLEBACK", "I_FORM", "PISTOL", "EMPTY", "JUMBO", "WILDCAT")
PLAY_TYPES = ("pass", "rush")
BOX_COUNTS = (4, 5, 6, 7, 8)
PERSONNEL_TYPES = ("Standard",)

# Build action maps
# Phase 1: Offense picks formation (7 actions)
FORMATION_ACTIONS = list(FORMATIONS)

# Phase 2: Defense picks box count (5 actions)
DEFENSE_ACTIONS = []
for box in BOX_COUNTS:
    for personnel in PERSONNEL_TYPES:
        DEFENSE_ACTIONS.append((box, personnel))

# Phase 3: Offense picks play type (2 actions)
PLAY_TYPE_ACTIONS = list(PLAY_TYPES)


class NFLGame:
    """NFL Play-by-Play Game compatible with RLCard.
    
    Three-phase game per play:
    Phase 1: Offense picks formation
    Phase 2: Defense sees formation, picks box count
    Phase 3: Offense sees box count, picks pass/rush
    """
    
    def __init__(self, allow_step_back=False, data_path=None, use_simple_model=None, single_play=False):
        """Initialize NFL Game.
        
        Args:
            allow_step_back: Whether to support step_back for CFR
            data_path: Path to cleaned NFL data (optional)
            use_simple_model: If True, skip pandas and use fast simplified model.
                             If None, auto-detect based on allow_step_back.
            single_play: If True, game ends after one complete play (3 phases).
                        This dramatically reduces tree depth for CFR algorithms.
        """
        self.allow_step_back = allow_step_back
        self.single_play = single_play
        self.np_random = np.random.RandomState()
        
        # Action spaces per phase
        self.formation_actions = FORMATION_ACTIONS  # Phase 1
        self.defense_actions = DEFENSE_ACTIONS      # Phase 2
        self.play_type_actions = PLAY_TYPE_ACTIONS  # Phase 3
        
        self.num_formation_actions = len(FORMATION_ACTIONS)
        self.num_defense_actions = len(DEFENSE_ACTIONS)
        self.num_play_type_actions = len(PLAY_TYPE_ACTIONS)
        
        # Max actions for RLCard compatibility
        self.num_actions = max(
            self.num_formation_actions,
            self.num_defense_actions,
            self.num_play_type_actions
        )
        
        # Use simple model for CFR (step_back) since pandas is too slow
        if use_simple_model is None:
            self.use_simple_model = allow_step_back
        else:
            self.use_simple_model = use_simple_model
        
        # Load data engine for outcomes (only if not using simple model)
        self.play_data = None
        if not self.use_simple_model:
            self._load_data(data_path)
        else:
            print("Using simplified outcome model (fast mode for CFR)")
        
        # Game state
        self.players = None
        self.down = None
        self.ydstogo = None
        self.yardline = None
        self.current_player = None
        
        # Phase tracking
        self.phase = 0  # 0=formation, 1=defense, 2=play_type
        self.pending_formation = None
        self.pending_defense_action = None
        
        self.is_over_flag = False
        self.payoffs = [0, 0]
        
        # History for step_back
        self.history = []
    
    def _load_data(self, data_path):
        """Load historical play data for outcome sampling."""
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
        
        # Initial state: 1st & 10 at own 25
        self.down = 1
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
        
        self.ep_before = self._calculate_ep(self.down, self.ydstogo, self.yardline)
        
        state = self.get_state(self.current_player)
        return state, self.current_player
    
    def step(self, action):
        """Process an action from current player."""
        if self.allow_step_back:
            self._save_state()
        
        if self.phase == 0:
            # Phase 0: Offense picks formation
            self.pending_formation = action if isinstance(action, str) else self.formation_actions[action]
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
            
            yards = outcome['yards_gained']
            turnover = outcome.get('turnover', False)
            
            # Update state
            old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            
            if turnover:
                self.is_over_flag = True
                epa = -old_ep * 2
            elif self.yardline + yards >= 100:
                # Touchdown!
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
                # No first down
                self.yardline += yards
                self.down += 1
                self.ydstogo -= yards
                
                if self.down > 4:
                    self.is_over_flag = True
                    epa = -old_ep
                else:
                    new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
                    epa = new_ep - old_ep
            
            if self.yardline <= 0:
                self.is_over_flag = True
                epa = -2.0 - old_ep
            
            self.payoffs = [epa, -epa]
            
            # In single_play mode, always end after one play
            if self.single_play:
                self.is_over_flag = True
            
            # Reset for next play (if not over)
            self.phase = 0
            self.current_player = 0
            self.pending_formation = None
            self.pending_defense_action = None
        
        state = self.get_state(self.current_player)
        return state, self.current_player
    
    def _get_outcome(self, down, ydstogo, yardline, offense_action, defense_action):
        """Sample outcome from historical data."""
        formation, play_type = offense_action
        box_count, _ = defense_action
        
        if self.play_data is None:
            # Simplified model
            if play_type == "pass":
                base_yards = 7.0
                variance = 10.0
                int_prob = 0.02
            else:
                base_yards = 4.0
                variance = 3.0
                int_prob = 0.01
            
            # Box count effects
            if play_type == "rush":
                base_yards -= (box_count - 6) * 0.5
            else:
                base_yards += (box_count - 6) * 0.3
            
            yards = self.np_random.normal(base_yards, variance)
            yards = max(-10, min(yards, 50))
            turnover = self.np_random.random() < int_prob
            
            return {'yards_gained': yards, 'turnover': turnover}
        
        # Use real data
        if play_type == "pass":
            candidates = self.play_data[self.play_data['pass'] == 1]
        else:
            candidates = self.play_data[self.play_data['rush'] == 1]
        
        if 'offense_formation' in candidates.columns:
            form_matches = candidates[candidates['offense_formation'] == formation]
            if len(form_matches) > 10:
                candidates = form_matches
        
        if 'defenders_in_box' in candidates.columns:
            box_matches = candidates[candidates['defenders_in_box'] == box_count]
            if len(box_matches) > 10:
                candidates = box_matches
        
        if len(candidates) == 0:
            return {'yards_gained': 0, 'turnover': False}
        
        candidates = candidates.copy()
        yardline_100 = 100 - yardline
        
        candidates['similarity'] = 1.0 / (1.0 + 
            abs(candidates['down'] - down) +
            abs(candidates['ydstogo'] - ydstogo).clip(0, 20) +
            abs(candidates['yardline_100'] - yardline_100).clip(0, 50) * 0.5
        )
        
        sampled = candidates.sample(n=1, weights='similarity').iloc[0]
        
        yards = sampled.get('yards_gained', 0)
        turnover = sampled.get('interception', 0) == 1 or sampled.get('fumble', 0) == 1
        
        return {'yards_gained': float(yards), 'turnover': bool(turnover)}
    
    def _calculate_ep(self, down, ydstogo, yardline):
        """Calculate expected points."""
        base_ep = (yardline / 100) * 7
        
        if down == 1:
            modifier = 0.0
        elif down == 2:
            modifier = -0.2
        elif down == 3:
            modifier = -0.5 - (ydstogo - 5) * 0.05
        else:
            modifier = -1.0 - ydstogo * 0.1
        
        return max(-2, min(7, base_ep + modifier))
    
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
            'payoffs': self.payoffs.copy()
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
        return True
    
    def get_state(self, player_id):
        """Get state from perspective of player."""
        if player_id == 0:
            # Offense
            if self.phase == 0:
                # Picking formation - just see game state
                return {
                    'down': self.down,
                    'ydstogo': self.ydstogo,
                    'yardline': self.yardline,
                    'phase': 'formation',
                    'legal_actions': list(range(self.num_formation_actions)),
                    'player_id': 0
                }
            else:
                # Phase 2: Picking play type - see box count!
                box_count = self.pending_defense_action[0] if self.pending_defense_action else 0
                return {
                    'down': self.down,
                    'ydstogo': self.ydstogo,
                    'yardline': self.yardline,
                    'formation': self.pending_formation,
                    'box_count': box_count,
                    'phase': 'play_type',
                    'legal_actions': list(range(self.num_play_type_actions)),
                    'player_id': 0
                }
        else:
            # Defense - sees formation
            return {
                'down': self.down,
                'ydstogo': self.ydstogo,
                'yardline': self.yardline,
                'formation': self.pending_formation,
                'phase': 'defense',
                'legal_actions': list(range(self.num_defense_actions)),
                'player_id': 1
            }
    
    def get_legal_actions(self):
        """Get legal actions for current player."""
        if self.phase == 0:
            return list(range(self.num_formation_actions))
        elif self.phase == 1:
            return list(range(self.num_defense_actions))
        else:
            return list(range(self.num_play_type_actions))
    
    def is_over(self):
        return self.is_over_flag
    
    def get_payoffs(self):
        return np.array(self.payoffs)
    
    def get_player_id(self):
        return self.current_player
    
    def get_num_players(self):
        return 2
    
    def get_num_actions(self):
        return self.num_actions
