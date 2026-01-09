"""
NFL Play-by-Play Game for RLCard

A two-player imperfect information game:
- Player 0 (Offense): Selects formation + play type
- Player 1 (Defense): Sees formation, selects box count + personnel

Game flow per play:
1. Offense selects action (formation, play_type)
2. Defense sees formation, selects (box_count, personnel)  
3. Outcome sampled from historical data
4. State updates, repeat until drive ends
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
OFFENSE_ACTIONS = []
for formation in FORMATIONS:
    for play_type in PLAY_TYPES:
        OFFENSE_ACTIONS.append((formation, play_type))

DEFENSE_ACTIONS = []
for box in BOX_COUNTS:
    for personnel in PERSONNEL_TYPES:
        DEFENSE_ACTIONS.append((box, personnel))


class NFLGame:
    """NFL Play-by-Play Game compatible with RLCard.
    
    This is an imperfect information game where:
    - Offense picks (formation, play_type) but defense only sees formation
    - Defense then picks (box_count, personnel) 
    - Outcome is sampled from historical NFL data
    """
    
    def __init__(self, allow_step_back=False, data_path=None):
        """Initialize NFL Game.
        
        Args:
            allow_step_back: Whether to support step_back for CFR
            data_path: Path to cleaned NFL data (optional)
        """
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        
        # Action spaces
        self.offense_actions = OFFENSE_ACTIONS
        self.defense_actions = DEFENSE_ACTIONS
        self.num_offense_actions = len(OFFENSE_ACTIONS)
        self.num_defense_actions = len(DEFENSE_ACTIONS)
        
        # Load data engine for outcomes
        self.data_engine = None
        self._load_data(data_path)
        
        # Game state
        self.players = None
        self.down = None
        self.ydstogo = None
        self.yardline = None
        self.current_player = None
        self.pending_offense_action = None
        self.is_over_flag = False
        self.payoffs = [0, 0]
        
        # History for step_back
        self.history = []
    
    def _load_data(self, data_path):
        """Load historical play data for outcome sampling."""
        if data_path is None:
            # Try to find data relative to this file
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
        pass  # No special configuration needed
    
    def init_game(self):
        """Initialize a new game (drive).
        
        Returns:
            (state, player_id): Initial state and starting player (0=offense)
        """
        # Create players
        self.players = [OffensePlayer(0), DefensePlayer(1)]
        
        # Initial state: 1st & 10 at own 25
        self.down = 1
        self.ydstogo = 10
        self.yardline = 25  # Yards from own goal
        
        # Offense goes first
        self.current_player = 0
        self.pending_offense_action = None
        self.is_over_flag = False
        self.payoffs = [0, 0]
        
        # Clear history
        self.history = []
        
        # Calculate initial EP
        self.ep_before = self._calculate_ep(self.down, self.ydstogo, self.yardline)
        
        state = self.get_state(self.current_player)
        return state, self.current_player
    
    def step(self, action):
        """Process an action from current player.
        
        Args:
            action: Action index or tuple
            
        Returns:
            (next_state, next_player_id)
        """
        if self.allow_step_back:
            self._save_state()
        
        if self.current_player == 0:
            # Offense plays
            self.pending_offense_action = action if isinstance(action, tuple) else self.offense_actions[action]
            # Now defense's turn
            self.current_player = 1
            state = self.get_state(self.current_player)
            return state, self.current_player
        
        else:
            # Defense plays, resolve the play
            defense_action = action if isinstance(action, tuple) else self.defense_actions[action]
            
            # Get outcome
            outcome = self._get_outcome(
                self.down, self.ydstogo, self.yardline,
                self.pending_offense_action, defense_action
            )
            
            yards = outcome['yards_gained']
            turnover = outcome.get('turnover', False)
            
            # Update state
            old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            
            if turnover:
                # Drive ends, offense loses
                self.is_over_flag = True
                new_ep = -old_ep  # Approximate turnover value
                epa = new_ep - old_ep
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
                    # Turnover on downs
                    self.is_over_flag = True
                    epa = -old_ep
                else:
                    new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
                    epa = new_ep - old_ep
            
            # Safety check
            if self.yardline <= 0:
                self.is_over_flag = True
                epa = -2.0 - old_ep
            
            # Update payoffs
            self.payoffs = [epa, -epa]  # Zero-sum
            
            # Back to offense for next play
            self.pending_offense_action = None
            self.current_player = 0
            
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
            
            # Adjust for box count
            if play_type == "rush":
                base_yards -= (box_count - 6) * 0.5
            else:
                base_yards += (box_count - 6) * 0.3
            
            yards = self.np_random.normal(base_yards, variance)
            yards = max(-10, min(yards, 50))
            turnover = self.np_random.random() < int_prob
            
            return {'yards_gained': yards, 'turnover': turnover}
        
        # Use real data - filter by play type using boolean columns
        if play_type == "pass":
            candidates = self.play_data[self.play_data['pass'] == 1]
        else:
            candidates = self.play_data[self.play_data['rush'] == 1]
        
        # Filter by formation if column exists
        if 'offense_formation' in candidates.columns:
            form_matches = candidates[candidates['offense_formation'] == formation]
            if len(form_matches) > 10:
                candidates = form_matches
        
        # Filter by box count if column exists
        if 'defenders_in_box' in candidates.columns:
            box_matches = candidates[candidates['defenders_in_box'] == box_count]
            if len(box_matches) > 10:
                candidates = box_matches
        
        if len(candidates) == 0:
            return {'yards_gained': 0, 'turnover': False}
        
        # Weight by similarity
        candidates = candidates.copy()
        
        # Use yardline_100 (yards from opponent's goal, so 100 - our yardline)
        yardline_100 = 100 - yardline
        
        candidates['similarity'] = 1.0 / (1.0 + 
            abs(candidates['down'] - down) +
            abs(candidates['ydstogo'] - ydstogo).clip(0, 20) +
            abs(candidates['yardline_100'] - yardline_100).clip(0, 50) * 0.5
        )
        
        # Sample
        sampled = candidates.sample(n=1, weights='similarity').iloc[0]
        
        yards = sampled.get('yards_gained', 0)
        turnover = sampled.get('interception', 0) == 1 or sampled.get('fumble', 0) == 1
        
        return {'yards_gained': float(yards), 'turnover': bool(turnover)}
    
    def _calculate_ep(self, down, ydstogo, yardline):
        """Calculate expected points from this situation."""
        # Simplified EP model
        # Base EP increases as you get closer to goal
        base_ep = (yardline / 100) * 7
        
        # Adjust for down and distance
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
            'pending_offense_action': self.pending_offense_action,
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
        self.pending_offense_action = state['pending_offense_action']
        self.is_over_flag = state['is_over_flag']
        self.payoffs = state['payoffs']
        return True
    
    def get_state(self, player_id):
        """Get state from perspective of player.
        
        Args:
            player_id: 0 for offense, 1 for defense
            
        Returns:
            State dictionary
        """
        if player_id == 0:
            # Offense doesn't see defense alignment
            return self.players[0].get_state(
                self.down, self.ydstogo, self.yardline,
                list(range(self.num_offense_actions))
            )
        else:
            # Defense sees formation
            formation = self.pending_offense_action[0] if self.pending_offense_action else "UNKNOWN"
            return self.players[1].get_state(
                self.down, self.ydstogo, self.yardline, formation,
                list(range(self.num_defense_actions))
            )
    
    def get_legal_actions(self):
        """Get legal actions for current player."""
        if self.current_player == 0:
            return list(range(self.num_offense_actions))
        else:
            return list(range(self.num_defense_actions))
    
    def is_over(self):
        """Check if game (drive) is over."""
        return self.is_over_flag
    
    def get_payoffs(self):
        """Get payoffs for both players."""
        return np.array(self.payoffs)
    
    def get_player_id(self):
        """Get current player."""
        return self.current_player
    
    def get_num_players(self):
        """Number of players (2: offense, defense)."""
        return 2
    
    def get_num_actions(self):
        """Number of actions (max across players)."""
        return max(self.num_offense_actions, self.num_defense_actions)
