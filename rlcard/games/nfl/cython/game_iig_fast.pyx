# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Fast Cython implementation of NFL IIG (Imperfect Information Game).

This is the "huddle commit" version where offense commits to
formation AND play type before seeing defense's response.

Key difference from regular game:
- Phase 0: Offense picks (Formation × PlayType) = 12 actions
- Phase 1: Defense sees Formation only, picks box count = 5 actions
- Phase 2: Auto-execute (no player action)

5-10x speedup for tabular CFR agents.
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
DEF NUM_PLAY_TYPES = 2
DEF NUM_SPECIAL_TEAMS = 2
DEF NUM_OFFENSE_ACTIONS = 12  # 5 formations × 2 play types + 2 special teams
DEF NUM_DEFENSE_ACTIONS = 5  # box counts 4-8
DEF NUM_PLAYERS = 2

# EP model coefficients
DEF EP_CONST = 0.2412
DEF EP_COEF_YARDLINE = 0.0571
DEF EP_COEF_YARDLINE_SQ = 5.853e-05
DEF EP_COEF_YDSTOGO = -0.0634
DEF EP_COEF_2ND_DOWN = -0.5528
DEF EP_COEF_3RD_DOWN = -1.2497
DEF EP_COEF_4TH_DOWN = -2.4989
DEF EP_COEF_REDZONE = -0.0255
DEF EP_COEF_GOAL_TO_GO = -0.1034


cdef class NFLGameIIGFast:
    """Fast Cython IIG NFL game implementation."""
    
    # Game state
    cdef public int down
    cdef public int ydstogo
    cdef public int yardline
    cdef public int phase
    cdef public int current_player
    cdef public int pending_formation_idx
    cdef public int pending_play_type  # 0=pass, 1=rush (hidden from defense)
    cdef public int pending_box_count
    cdef public bint is_over_flag
    cdef public bint single_play
    cdef public double[:] payoffs
    
    # Outcome sampling
    cdef object outcome_model
    cdef object np_random
    cdef object special_teams
    
    def __init__(self, bint single_play=True, object outcome_model=None,
                 object special_teams=None, seed=None):
        """Initialize fast IIG NFL game."""
        self.single_play = single_play
        self.outcome_model = outcome_model
        self.special_teams = special_teams
        self.payoffs = np.zeros(NUM_PLAYERS, dtype=np.float64)
        
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()
    
    cpdef tuple init_game(self):
        """Initialize new game."""
        self.down = 1
        self.ydstogo = 10
        self.yardline = 25
        self.phase = 0
        self.current_player = 0
        self.pending_formation_idx = -1
        self.pending_play_type = -1
        self.pending_box_count = 0
        self.is_over_flag = False
        self.payoffs[0] = 0.0
        self.payoffs[1] = 0.0
        
        return self._get_state_dict(), self.current_player
    
    cdef inline double _calculate_ep(self, int down, int ydstogo, int yardline,
                                      bint goal_to_go=False) noexcept:
        """Fast EP calculation."""
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
        
        if ep < -7.0:
            return -7.0
        if ep > 7.0:
            return 7.0
        return ep
    
    cpdef tuple step(self, int action):
        """Process action."""
        cdef double old_ep, epa
        cdef int formation_idx, play_type, yards
        cdef bint turnover
        
        if self.phase == 0:
            # Offense commits to (Formation, PlayType) or special teams
            if action >= NUM_FORMATIONS * NUM_PLAY_TYPES:
                # Special teams (10=PUNT, 11=FG)
                self._resolve_special_teams(action - NUM_FORMATIONS * NUM_PLAY_TYPES)
            else:
                # Normal play - decode formation and play type
                # Actions 0-1: SHOTGUN pass/rush
                # Actions 2-3: SINGLEBACK pass/rush
                # etc.
                formation_idx = action // NUM_PLAY_TYPES
                play_type = action % NUM_PLAY_TYPES  # 0=pass, 1=rush
                
                self.pending_formation_idx = formation_idx
                self.pending_play_type = play_type
                self.phase = 1
                self.current_player = 1
        
        elif self.phase == 1:
            # Defense picks box count (sees formation, NOT play type)
            self.pending_box_count = action + 4
            
            # Auto-execute the committed play
            old_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline)
            yards, turnover = self._sample_outcome()
            self._apply_outcome(yards, turnover, old_ep)
            
            # Reset for next play
            self.phase = 0
            self.current_player = 0
            self.pending_formation_idx = -1
            self.pending_play_type = -1
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
                opp_yardline = max(20, 100 - (self.yardline + 40))
            
            opp_yardline = max(1, min(99, opp_yardline))
            opp_ep = self._calculate_ep(1, 10, opp_yardline)
            epa = -opp_ep - old_ep
        
        self.is_over_flag = True
        self.payoffs[0] = epa
        self.payoffs[1] = -epa
        self.phase = 0
        self.current_player = 0
    
    cdef tuple _sample_outcome(self):
        """Sample outcome using committed play type."""
        cdef int yards
        cdef bint turnover
        cdef double base_yards, variance, int_prob
        
        if self.outcome_model is not None:
            formation = ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY'][self.pending_formation_idx]
            pt = 'pass' if self.pending_play_type == 0 else 'rush'
            result = self.outcome_model.sample(
                formation, pt, self.pending_box_count,
                self.yardline, self.down, self.ydstogo
            )
            return int(result['yards_gained']), result.get('turnover', False)
        
        # Simple fallback
        if self.pending_play_type == 0:  # pass
            base_yards = 7.0
            variance = 10.0
            int_prob = 0.02
        else:  # rush
            base_yards = 4.0
            variance = 3.0
            int_prob = 0.01
        
        base_yards -= (self.pending_box_count - 6) * (0.5 if self.pending_play_type == 1 else -0.3)
        
        yards = int(self.np_random.normal(base_yards, variance))
        yards = max(-10, min(yards, 50))
        turnover = self.np_random.random() < int_prob
        
        return yards, turnover
    
    cdef void _apply_outcome(self, int yards, bint turnover, double old_ep):
        """Apply outcome and calculate EPA."""
        cdef double new_ep, opp_ep, epa
        cdef int opp_yardline
        cdef bint is_goal_to_go
        
        if turnover:
            self.is_over_flag = True
            opp_yardline = 100 - (self.yardline + yards)
            opp_yardline = max(1, min(99, opp_yardline))
            opp_ep = self._calculate_ep(1, 10, opp_yardline)
            epa = -opp_ep - old_ep
        
        elif self.yardline + yards >= 100:
            self.is_over_flag = True
            epa = 7.0 - old_ep
        
        elif yards >= self.ydstogo:
            self.yardline += yards
            self.down = 1
            self.ydstogo = 10 if 100 - self.yardline >= 10 else 100 - self.yardline
            is_goal_to_go = (100 - self.yardline) < self.ydstogo
            new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline, is_goal_to_go)
            epa = new_ep - old_ep
        
        else:
            self.yardline += yards
            self.down += 1
            self.ydstogo -= yards
            
            if self.down > 4:
                self.is_over_flag = True
                opp_yardline = 100 - self.yardline
                opp_yardline = max(1, min(99, opp_yardline))
                opp_ep = self._calculate_ep(1, 10, opp_yardline)
                epa = -opp_ep - old_ep
            else:
                is_goal_to_go = (100 - self.yardline) < self.ydstogo
                new_ep = self._calculate_ep(self.down, self.ydstogo, self.yardline, is_goal_to_go)
                epa = new_ep - old_ep
        
        if self.yardline <= 0:
            self.is_over_flag = True
            opp_ep = self._calculate_ep(1, 10, 35)
            epa = -(2.0 + opp_ep) - old_ep
        
        if self.single_play:
            self.is_over_flag = True
        
        self.payoffs[0] = epa
        self.payoffs[1] = -epa
    
    cdef dict _get_state_dict(self):
        """Get state as dict. Defense does NOT see pending_play_type."""
        cdef list legal_actions
        
        if self.phase == 0:
            legal_actions = list(range(NUM_OFFENSE_ACTIONS))
        elif self.phase == 1:
            legal_actions = list(range(NUM_DEFENSE_ACTIONS))
        else:
            legal_actions = []
        
        return {
            'down': self.down,
            'ydstogo': self.ydstogo,
            'yardline': self.yardline,
            'phase': self.phase,
            'formation_idx': self.pending_formation_idx,  # Visible to defense
            # Note: pending_play_type is NOT included - hidden from defense
            'legal_actions': legal_actions,
            'player_id': self.current_player,
        }
    
    cpdef bint is_over(self):
        return self.is_over_flag
    
    cpdef np.ndarray get_payoffs(self):
        return np.asarray(self.payoffs)
    
    cpdef list get_legal_actions(self):
        if self.phase == 0:
            return list(range(NUM_OFFENSE_ACTIONS))
        elif self.phase == 1:
            return list(range(NUM_DEFENSE_ACTIONS))
        return []
    
    cpdef int get_player_id(self):
        return self.current_player
    
    cpdef int get_num_players(self):
        return NUM_PLAYERS
    
    cpdef int get_num_actions(self):
        return NUM_OFFENSE_ACTIONS


def make_fast_iig_game(single_play=True, use_cached_model=True, seed=None):
    """Factory to create fast IIG game."""
    try:
        from rlcard.games.nfl.cached_outcome_model import get_cached_outcome_model
        from rlcard.games.nfl.special_teams import get_special_teams_engine
        import pandas as pd
        from pathlib import Path
        
        data_path = Path(__file__).parent.parent.parent.parent.parent / "Code" / "data" / "cleaned_nfl_rl_data.csv"
        if data_path.exists():
            play_data = pd.read_csv(data_path)
            np_random = np.random.RandomState(seed) if seed else np.random.RandomState()
            outcome_model = get_cached_outcome_model(play_data, np_random, use_bucketed=True)
        else:
            outcome_model = None
        
        special_teams = get_special_teams_engine()
        
        return NFLGameIIGFast(
            single_play=single_play,
            outcome_model=outcome_model if use_cached_model else None,
            special_teams=special_teams,
            seed=seed
        )
    except Exception as e:
        print(f"Cython IIG game unavailable: {e}")
        from rlcard.games.nfl.game_iig import NFLGameIIG
        return NFLGameIIG(single_play=single_play, use_cached_model=use_cached_model)
