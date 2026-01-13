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
from rlcard.games.nfl.special_teams import get_special_teams_engine


# Game constants - these should match your cleaned_nfl_rl_data.csv
FORMATIONS = ("SHOTGUN", "SINGLEBACK", "UNDER CENTER", "I_FORM", "EMPTY")
PLAY_TYPES = ("pass", "rush")
BOX_COUNTS = (4, 5, 6, 7, 8)
PERSONNEL_TYPES = ("Standard",)
SPECIAL_TEAMS = ("PUNT", "FG")  # These bypass defense

# Build action maps
# Phase 0: Offense picks formation OR special teams (7 actions total)
FORMATION_ACTIONS = list(FORMATIONS)
SPECIAL_TEAMS_ACTIONS = list(SPECIAL_TEAMS)
INITIAL_ACTIONS = FORMATION_ACTIONS + SPECIAL_TEAMS_ACTIONS  # All Phase 0 options

# Phase 1: Defense picks box count (5 actions)
DEFENSE_ACTIONS = []
for box in BOX_COUNTS:
    for personnel in PERSONNEL_TYPES:
        DEFENSE_ACTIONS.append((box, personnel))

# Phase 2: Offense picks play type (2 actions)
PLAY_TYPE_ACTIONS = list(PLAY_TYPES)


class NFLGame:
    """NFL Play-by-Play Game compatible with RLCard.
    
    Three-phase game per play:
    Phase 1: Offense picks formation
    Phase 2: Defense sees formation, picks box count
    Phase 3: Offense sees box count, picks pass/rush
    """
    
    def __init__(self, allow_step_back=False, data_path=None, use_simple_model=None, 
                 single_play=False, use_distribution_model=False):
        """Initialize NFL Game.
        
        Args:
            allow_step_back: Whether to support step_back for CFR
            data_path: Path to cleaned NFL data (optional)
            use_simple_model: If True, skip pandas and use fast simplified model.
                             If None, auto-detect based on allow_step_back.
            single_play: If True, game ends after one complete play (3 phases).
                        This dramatically reduces tree depth for CFR algorithms.
            use_distribution_model: If True, use Biro & Walker statistical distributions
                        instead of random sampling for play outcomes.
        """
        self.allow_step_back = allow_step_back
        self.single_play = single_play
        self.use_distribution_model = use_distribution_model
        self.np_random = np.random.RandomState()
        
        # Action spaces per phase
        self.initial_actions = INITIAL_ACTIONS          # Phase 0: formations + special teams
        self.formation_actions = FORMATION_ACTIONS      # Subset of phase 0
        self.special_teams_actions = SPECIAL_TEAMS_ACTIONS  # Subset of phase 0
        self.defense_actions = DEFENSE_ACTIONS          # Phase 1
        self.play_type_actions = PLAY_TYPE_ACTIONS      # Phase 2
        
        self.num_initial_actions = len(INITIAL_ACTIONS)
        self.num_formation_actions = len(FORMATION_ACTIONS)
        self.num_defense_actions = len(DEFENSE_ACTIONS)
        self.num_play_type_actions = len(PLAY_TYPE_ACTIONS)
        
        # Max actions for RLCard compatibility
        self.num_actions = max(
            self.num_initial_actions,  # 7 (5 formations + 2 special teams)
            self.num_defense_actions,  # 5 box counts
            self.num_play_type_actions # 2 play types
        )
        
        # Special teams engine for FG/Punt outcomes
        self.special_teams = get_special_teams_engine()
        
        # Use simple model for CFR (step_back) since pandas is too slow
        # UNLESS distribution model is enabled (which is efficient)
        if use_simple_model is None:
            self.use_simple_model = allow_step_back and not use_distribution_model
        else:
            self.use_simple_model = use_simple_model
        
        # Load data engine for outcomes (only if not using simple model OR using distribution model)
        self.play_data = None
        self.outcome_model = None
        if not self.use_simple_model or self.use_distribution_model:
            self._load_data(data_path)
            # Initialize statistical outcome model if enabled
            if self.use_distribution_model and self.play_data is not None:
                from rlcard.games.nfl.outcome_model import OutcomeModel
                self.outcome_model = OutcomeModel(self.play_data, self.np_random)
                print("Using Biro & Walker distribution model for outcomes")
            elif not self.use_simple_model:
                pass  # Will use data-based sampling
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
            # Phase 0: Offense picks formation OR special teams
            action_str = action if isinstance(action, str) else self.initial_actions[action]
            
            if action_str in SPECIAL_TEAMS:
                # Special teams - skip defense, resolve immediately
                self._resolve_special_teams(action_str)
            else:
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
            
            yards = outcome['yards_gained']
            turnover = outcome.get('turnover', False)
            
            # Update state
            old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            
            if turnover:
                self.is_over_flag = True
                # Estimate opponent EP from turnover spot
                turnover_spot = self.yardline + yards
                turnover_spot = max(1, min(99, turnover_spot))
                opp_yardline = 100 - turnover_spot
                opp_ep = self._calculate_ep(down=1, ydstogo=10, yardline=opp_yardline)
                epa = -opp_ep - old_ep
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
                    # Turnover on downs: opponent gets ball here
                    # Calculate opponent's EP from their new field position
                    opp_yardline = 100 - self.yardline  # Convert to opponent's perspective
                    opp_yardline = max(1, min(99, opp_yardline))
                    opp_ep = self._calculate_ep(down=1, ydstogo=10, yardline=opp_yardline)
                    epa = -opp_ep - old_ep
                    self.is_over_flag = True
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
    
    def _resolve_special_teams(self, action_str):
        """Resolve a special teams play (punt or field goal).
        
        Special teams plays skip the defense turn and resolve immediately.
        """
        old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
        
        if action_str == "FG":
            # Field Goal attempt
            success_prob = self.special_teams.predict_fg_prob(self.yardline)
            success = self.np_random.random() < success_prob
            
            if success:
                # FG made: +3 points
                epa = 3.0 - old_ep
            else:
                # FG missed: opponent gets ball at LOS or 20 (their yardline)
                opp_yardline = max(100 - self.yardline, 20)
                # Use full EP model for opponent's position (1st & 10)
                opp_ep = self._calculate_ep(down=1, ydstogo=10, yardline=opp_yardline)
                epa = -opp_ep - old_ep
            
            self.is_over_flag = True
            
        elif action_str == "PUNT":
            # Punt: opponent gets ball at predicted position
            # predict_punt_outcome returns opponent's yardline from THEIR own goal
            opp_yardline = self.special_teams.predict_punt_outcome(self.yardline)
            opp_yardline = max(1, min(99, opp_yardline))  # Clamp to valid range
            
            # Use full EP model for opponent's position (1st & 10)
            opp_ep = self._calculate_ep(down=1, ydstogo=10, yardline=opp_yardline)
            epa = -opp_ep - old_ep
            
            self.is_over_flag = True
        else:
            epa = 0
        
        self.payoffs = [epa, -epa]
        
        # Always end after special teams (no continuation)
        self.phase = 0
        self.current_player = 0
        self.pending_formation = None
        self.pending_defense_action = None
    
    def _get_outcome(self, down, ydstogo, yardline, offense_action, defense_action):
        """Sample outcome from historical data or statistical distribution."""
        formation, play_type = offense_action
        box_count, _ = defense_action
        
        # Use distribution model if enabled and available
        if self.outcome_model is not None:
            return self.outcome_model.get_outcome(
                down, ydstogo, yardline, formation, box_count, play_type
            )
        
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
    
    def _calculate_ep(self, down, ydstogo, yardline, goal_to_go=False):
        """Calculate expected points using fitted OLS regression model.
        
        Model: EP = const + sum(coef_i * feature_i)
        R² = 0.95 on 210,490 NFL plays
        
        Args:
            down: Current down (1-4)
            ydstogo: Yards to first down
            yardline: Yards from own goal (1-99)
            goal_to_go: Whether it's a goal-to-go situation
        
        Returns:
            Expected points (-7 to 7 range)
        """
        # Fitted coefficients from OLS regression
        # See examples/fit_ep_model.py for derivation
        CONST = 0.2412
        COEF_YARDLINE = 0.0571
        COEF_YARDLINE_SQ = 5.853e-05
        COEF_YDSTOGO = -0.0634
        COEF_2ND_DOWN = -0.5528
        COEF_3RD_DOWN = -1.2497
        COEF_4TH_DOWN = -2.4989
        COEF_REDZONE = -0.0255
        COEF_GOAL_TO_GO = -0.1034
        # score_diff coefficient is ~0 and not significant, so excluded
        
        # Build prediction
        ep = CONST
        ep += COEF_YARDLINE * yardline
        ep += COEF_YARDLINE_SQ * (yardline ** 2)
        ep += COEF_YDSTOGO * min(ydstogo, 20)  # Cap at 20 as in training
        
        # Down dummies (reference = 1st down)
        if down == 2:
            ep += COEF_2ND_DOWN
        elif down == 3:
            ep += COEF_3RD_DOWN
        elif down == 4:
            ep += COEF_4TH_DOWN
        
        # Red zone indicator
        if yardline >= 80:
            ep += COEF_REDZONE
        
        # Goal-to-go
        if goal_to_go:
            ep += COEF_GOAL_TO_GO
        
        return max(-7, min(7, ep))
    
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
                # Picking formation or special teams - just see game state
                return {
                    'down': self.down,
                    'ydstogo': self.ydstogo,
                    'yardline': self.yardline,
                    'phase': 'formation',
                    'legal_actions': list(range(self.num_initial_actions)),  # 5 formations + 2 special teams
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
            return list(range(self.num_initial_actions))  # Formations + special teams
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
