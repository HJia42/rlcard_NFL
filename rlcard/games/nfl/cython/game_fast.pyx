# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fast Cython implementation of NFL Game.

This provides 5-10x speedup for tabular CFR agents by using:
- C-typed variables and arrays
- Inline binning functions
- Minimal Python object creation

Usage:
    from rlcard.games.nfl.cython.game_fast import NFLGameFast
    game = NFLGameFast(single_play=True)
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sqrt

# Type definitions
DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

# Game constants
DEF NUM_FORMATIONS = 5
DEF NUM_SPECIAL_TEAMS = 2
DEF NUM_INITIAL_ACTIONS = 7  # 5 formations + 2 special teams
DEF NUM_DEFENSE_ACTIONS = 5  # box counts 4-8
DEF NUM_PLAY_TYPES = 2  # pass, rush
DEF NUM_PLAYERS = 2

# EP model coefficients (from OLS regression)
DEF EP_CONST = 0.2412
DEF EP_COEF_YARDLINE = 0.0571
DEF EP_COEF_YARDLINE_SQ = 5.853e-05
DEF EP_COEF_YDSTOGO = -0.0634
DEF EP_COEF_2ND_DOWN = -0.5528
DEF EP_COEF_3RD_DOWN = -1.2497
DEF EP_COEF_4TH_DOWN = -2.4989
DEF EP_COEF_REDZONE = -0.0255
DEF EP_COEF_GOAL_TO_GO = -0.1034


cdef class NFLGameFast:
    """Fast Cython NFL game implementation."""
    
    # Game state (C-typed for speed)
    cdef public int down
    cdef public int ydstogo
    cdef public int yardline
    cdef public int phase
    cdef public int current_player
    cdef public int pending_formation_idx
    cdef public int pending_box_count
    cdef public bint is_over_flag
    cdef public bint single_play
    cdef public double[:] payoffs
    
    # Outcome sampling (cached)
    cdef object outcome_model
    cdef object np_random
    cdef object special_teams
    
    def __init__(self, bint single_play=True, object outcome_model=None, 
                 object special_teams=None, seed=None):
        """Initialize fast NFL game.
        
        Args:
            single_play: End after one play (for CFR)
            outcome_model: CachedOutcomeModel for sampling
            special_teams: SpecialTeamsEngine for FG/punt
            seed: Random seed
        """
        self.single_play = single_play
        self.outcome_model = outcome_model
        self.special_teams = special_teams
        self.payoffs = np.zeros(NUM_PLAYERS, dtype=np.float64)
        
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
    
    cpdef tuple init_game(self):
        """Initialize new game. Returns (state, player_id)."""
        self.down = 1
        self.ydstogo = 10
        self.yardline = 25
        self.phase = 0
        self.current_player = 0
        self.pending_formation_idx = -1
        self.pending_box_count = 0
        self.is_over_flag = False
        self.payoffs[0] = 0.0
        self.payoffs[1] = 0.0
        
        return self._get_state_dict(), self.current_player
    
    cdef inline double _calculate_ep(self, int down, int ydstogo, int yardline, 
                                      bint goal_to_go=False) noexcept:
        """Fast EP calculation with C types."""
        cdef double ep = EP_CONST
        cdef int capped_ydstogo = ydstogo if ydstogo < 20 else 20
        
        ep += EP_COEF_YARDLINE * yardline
        ep += EP_COEF_YARDLINE_SQ * (yardline * yardline)
        ep += EP_COEF_YDSTOGO * capped_ydstogo
        
        if down == 2:
            ep += EP_COEF_2ND_DOWN
        elif down == 3:
            ep += EP_COEF_3RD_DOWN
        elif down == 4:
            ep += EP_COEF_4TH_DOWN
        
        if yardline >= 80:
            ep += EP_COEF_REDZONE
        
        if goal_to_go:
            ep += EP_COEF_GOAL_TO_GO
        
        # Clamp to [-7, 7]
        if ep < -7.0:
            return -7.0
        if ep > 7.0:
            return 7.0
        return ep
    
    cpdef tuple step(self, int action):
        """Process action. Returns (state, player_id)."""
        cdef double old_ep, new_ep, epa, opp_ep
        cdef int yards
        cdef bint turnover, is_goal_to_go
        
        if self.phase == 0:
            # Offense picks formation or special teams
            if action >= NUM_FORMATIONS:
                # Special teams (PUNT=5, FG=6)
                self._resolve_special_teams(action - NUM_FORMATIONS)
            else:
                # Normal play - go to defense phase
                self.pending_formation_idx = action
                self.phase = 1
                self.current_player = 1
        
        elif self.phase == 1:
            # Defense picks box count (action 0-4 -> box 4-8)
            self.pending_box_count = action + 4
            self.phase = 2
            self.current_player = 0
        
        elif self.phase == 2:
            # Offense picks play type, resolve outcome
            old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            
            # Get outcome (yards, turnover)
            yards, turnover = self._sample_outcome(action)
            
            if turnover:
                self.is_over_flag = True
                opp_ep = self._calculate_ep(1, 10, 100 - (self.yardline + yards))
                epa = -opp_ep - old_ep
            
            elif self.yardline + yards >= 100:
                # Touchdown
                self.is_over_flag = True
                epa = 7.0 - old_ep
            
            elif yards >= self.ydstogo:
                # First down
                self.yardline += yards
                self.down = 1
                self.ydstogo = 10 if 100 - self.yardline >= 10 else 100 - self.yardline
                is_goal_to_go = (100 - self.yardline) < self.ydstogo
                new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline, is_goal_to_go)
                epa = new_ep - old_ep
            
            else:
                # No first down
                self.yardline += yards
                self.down += 1
                self.ydstogo -= yards
                
                if self.down > 4:
                    # Turnover on downs
                    self.is_over_flag = True
                    opp_ep = self._calculate_ep(1, 10, 100 - self.yardline)
                    epa = -opp_ep - old_ep
                else:
                    is_goal_to_go = (100 - self.yardline) < self.ydstogo
                    new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline, is_goal_to_go)
                    epa = new_ep - old_ep
            
            # Safety check
            if self.yardline <= 0:
                self.is_over_flag = True
                opp_ep = self._calculate_ep(1, 10, 35)  # Free kick position
                epa = -(2.0 + opp_ep) - old_ep
            
            self.payoffs[0] = epa
            self.payoffs[1] = -epa
            
            if self.single_play:
                self.is_over_flag = True
            
            # Reset for next play
            self.phase = 0
            self.current_player = 0
            self.pending_formation_idx = -1
            self.pending_box_count = 0
        
        return self._get_state_dict(), self.current_player
    
    cdef void _resolve_special_teams(self, int st_action):
        """Resolve punt (0) or FG (1)."""
        cdef double old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
        cdef double opp_ep, epa, success_prob
        cdef int opp_yardline
        
        if st_action == 1:  # FG
            if self.special_teams is not None:
                success_prob = self.special_teams.predict_fg_prob(self.yardline)
            else:
                # Simple model
                success_prob = max(0, min(1, 0.95 - 0.027 * ((100 - self.yardline + 17) - 30)))
            
            if self.np_random.random() < success_prob:
                epa = 3.0 - old_ep
            else:
                opp_yardline = max(100 - self.yardline, 20)
                opp_ep = self._calculate_ep(1, 10, opp_yardline)
                epa = -opp_ep - old_ep
        
        else:  # PUNT
            if self.special_teams is not None:
                opp_yardline = int(self.special_teams.predict_punt_outcome(self.yardline))
            else:
                # Simple model: ~40 yard net
                opp_yardline = max(20, 100 - (self.yardline + 40))
            
            opp_yardline = max(1, min(99, opp_yardline))
            opp_ep = self._calculate_ep(1, 10, opp_yardline)
            epa = -opp_ep - old_ep
        
        self.is_over_flag = True
        self.payoffs[0] = epa
        self.payoffs[1] = -epa
        self.phase = 0
        self.current_player = 0
    
    cdef tuple _sample_outcome(self, int play_type):
        """Sample yards and turnover. Returns (yards, turnover)."""
        cdef int yards
        cdef bint turnover
        cdef double base_yards, variance, int_prob
        
        if self.outcome_model is not None:
            # Use cached model
            formation = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY'][self.pending_formation_idx]
            pt = 'pass' if play_type == 0 else 'rush'
            result = self.outcome_model.sample(
                formation, pt, self.pending_box_count,
                self.yardline, self.down, self.ydstogo
            )
            return int(result['yards_gained']), result.get('turnover', False)
        
        # Simple fallback model
        if play_type == 0:  # pass
            base_yards = 7.0
            variance = 10.0
            int_prob = 0.02
        else:  # rush
            base_yards = 4.0
            variance = 3.0
            int_prob = 0.01
        
        # Box count effect
        if play_type == 1:  # rush
            base_yards -= (self.pending_box_count - 6) * 0.5
        else:
            base_yards += (self.pending_box_count - 6) * 0.3
        
        yards = int(self.np_random.normal(base_yards, variance))
        yards = max(-10, min(yards, 50))
        turnover = self.np_random.random() < int_prob
        
        return yards, turnover
    
    cdef dict _get_state_dict(self):
        """Get state as Python dict for RLCard compatibility.
        
        Matches standardized format from Python games:
        - obs: 12-dim float32 array
        - legal_actions: dict {action_idx: None}
        - raw_legal_actions: list of action names
        - phase: int (0/1/2)
        - phase_name: str
        """
        cdef list legal_list
        cdef list raw_legal_actions
        cdef str phase_name
        cdef np.ndarray[DTYPE_t, ndim=1] obs = np.zeros(12, dtype=DTYPE)
        
        # Formation names for raw_legal_actions
        formations = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY', 'PUNT', 'FG']
        defense_names = ['4_box', '5_box', '6_box', '7_box', '8_box']
        play_types = ['pass', 'rush']
        
        # Build obs array
        obs[0] = self.down / 4.0
        obs[1] = min(self.ydstogo, 30) / 30.0
        obs[2] = self.yardline / 100.0
        
        # Formation encoding (indices 3-7) - visible after phase 0
        if self.phase >= 1 and 0 <= self.pending_formation_idx < NUM_FORMATIONS:
            obs[3 + self.pending_formation_idx] = 1.0
        
        # Box count encoding (index 8) - visible in phase 2
        if self.phase == 2 and self.pending_box_count > 0:
            obs[8] = (self.pending_box_count - 4) / 4.0
        
        # Phase encoding
        obs[11] = self.phase / 2.0
        
        # Build legal actions and raw names based on phase
        if self.phase == 0:
            legal_list = list(range(NUM_INITIAL_ACTIONS))
            raw_legal_actions = formations
            phase_name = 'formation'
        elif self.phase == 1:
            legal_list = list(range(NUM_DEFENSE_ACTIONS))
            raw_legal_actions = defense_names
            phase_name = 'defense'
        else:
            legal_list = list(range(NUM_PLAY_TYPES))
            raw_legal_actions = play_types
            phase_name = 'play_type'
        
        # Convert to dict for RLCard compatibility
        legal_actions_dict = {i: None for i in legal_list}
        
        state = {
            'obs': obs,
            'legal_actions': legal_actions_dict,
            'raw_legal_actions': raw_legal_actions,
            'player_id': self.current_player,
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': self.phase,
            'phase_name': phase_name,
        }
        
        # Add phase-specific info
        if self.phase >= 1 and 0 <= self.pending_formation_idx < NUM_FORMATIONS:
            state['formation'] = formations[self.pending_formation_idx]
        if self.phase == 2 and self.pending_box_count > 0:
            state['box_count'] = self.pending_box_count
        
        return state
    
    cpdef bint is_over(self):
        """Check if game is over."""
        return self.is_over_flag
    
    cpdef np.ndarray get_payoffs(self):
        """Get payoffs array."""
        return np.asarray(self.payoffs)
    
    cpdef list get_legal_actions(self):
        """Get legal actions for current phase."""
        if self.phase == 0:
            return list(range(NUM_INITIAL_ACTIONS))
        elif self.phase == 1:
            return list(range(NUM_DEFENSE_ACTIONS))
        else:
            return list(range(NUM_PLAY_TYPES))
    
    cpdef int get_player_id(self):
        """Get current player ID."""
        return self.current_player
    
    cpdef int get_num_players(self):
        """Get number of players."""
        return NUM_PLAYERS
    
    cpdef int get_num_actions(self):
        """Get max number of actions."""
        return NUM_INITIAL_ACTIONS


def make_fast_game(single_play=True, use_cached_model=True, use_bucketed=True, seed=None):
    """Factory to create fast game with optional cached outcome model.
    
    Args:
        single_play: End game after one play
        use_cached_model: Use O(1) cached outcome model
        use_bucketed: Use bucketed or full state space for caching
        seed: Random seed
    
    Falls back to Python version if Cython not compiled.
    """
    try:
        from rlcard.games.nfl.cached_outcome_model import get_cached_outcome_model
        from rlcard.games.nfl.special_teams import get_special_teams_engine
        import pandas as pd
        from pathlib import Path
        
        # Load play data for outcome model
        data_path = Path(__file__).parent.parent.parent.parent.parent / "Code" / "data" / "cleaned_nfl_rl_data.csv"
        if data_path.exists():
            play_data = pd.read_csv(data_path)
            np_random = np.random.RandomState(seed) if seed else np.random.RandomState()
            outcome_model = get_cached_outcome_model(play_data, np_random, use_bucketed=use_bucketed)
        else:
            outcome_model = None
            np_random = None
        
        special_teams = get_special_teams_engine()
        
        return NFLGameFast(
            single_play=single_play,
            outcome_model=outcome_model if use_cached_model else None,
            special_teams=special_teams,
            seed=seed
        )
    except Exception as e:
        print(f"Cython game unavailable, falling back to Python: {e}")
        from rlcard.games.nfl.game import NFLGame
        return NFLGame(single_play=single_play, use_cached_model=use_cached_model)
