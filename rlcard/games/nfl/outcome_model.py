"""
Statistical Outcome Model for NFL Plays

Based on Biro & Walker (2021, 2023):
- Run plays: Skew-t distribution + TD point mass
- Pass plays: Gamma for completions, categorical for outcomes

This replaces random sampling with distribution-based outcomes for more
stable training and faster convergence.
"""

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func


# Biro & Walker league-average skew-t parameters for runs
SKEWT_DEFAULT = {
    'location': 0.221,
    'scale': 4.150,
    'skewness': 1.811,
    'df': 2.505,
    'td_weight': 0.003,  # 1 - 0.997 mixing weight
}

# Gamma parameters for pass completions
GAMMA_ALPHA = 1.3  # Fixed shape per Biro & Walker
GAMMA_PRIOR_ALPHA = 130
GAMMA_PRIOR_BETA = 1300

# Minimum plays threshold for direct fit vs KNN fallback
MIN_PLAYS_THRESHOLD = 100


class OutcomeModel:
    """Statistical model for NFL play outcomes."""
    
    def __init__(self, play_data, np_random=None):
        """Initialize outcome model with historical play data.
        
        Args:
            play_data: DataFrame with historical plays
            np_random: Numpy random state for reproducibility
        """
        self.play_data = play_data
        self.np_random = np_random or np.random.default_rng()
        
        # Cache fitted distributions by key
        self._cache = {}
    
    def get_outcome(self, down, ydstogo, yardline, formation, box_count, play_type):
        """Sample play outcome from fitted distribution.
        
        Args:
            down: Current down (1-4)
            ydstogo: Yards to first down
            yardline: Yards from own goal (1-99)
            formation: Offensive formation
            box_count: Defenders in box
            play_type: 'pass' or 'rush'
            
        Returns:
            dict with 'yards_gained' and 'turnover'
        """
        # Get matching plays
        candidates = self._get_candidates(formation, box_count, play_type)
        
        # Check if we have enough data
        if len(candidates) < MIN_PLAYS_THRESHOLD:
            # Use KNN-weighted sampling + distribution fit
            candidates = self._knn_expand(candidates, down, ydstogo, yardline, play_type)
        
        if len(candidates) < 10:
            # Fallback to simple model
            return self._simple_outcome(play_type, box_count)
        
        # Fit and sample from distribution
        if play_type == 'rush':
            return self._sample_run(candidates, yardline)
        else:
            return self._sample_pass(candidates, yardline)
    
    def _get_candidates(self, formation, box_count, play_type):
        """Get plays matching formation, box count, and play type."""
        if self.play_data is None:
            return []
        
        # Filter by play type
        if play_type == 'pass':
            candidates = self.play_data[self.play_data['pass'] == 1]
        else:
            candidates = self.play_data[self.play_data['rush'] == 1]
        
        # Filter by formation
        if 'offense_formation' in candidates.columns:
            form_matches = candidates[candidates['offense_formation'] == formation]
            if len(form_matches) > 20:
                candidates = form_matches
        
        # Filter by box count
        if 'defenders_in_box' in candidates.columns:
            box_matches = candidates[candidates['defenders_in_box'] == box_count]
            if len(box_matches) > 20:
                candidates = box_matches
        
        return candidates
    
    def _knn_expand(self, candidates, down, ydstogo, yardline, play_type):
        """Expand candidate pool using KNN similarity to game situation."""
        # Start from play type matches if candidates empty
        if len(candidates) < 10:
            if play_type == 'pass':
                candidates = self.play_data[self.play_data['pass'] == 1]
            else:
                candidates = self.play_data[self.play_data['rush'] == 1]
        
        if len(candidates) == 0:
            return candidates
        
        # Compute similarity scores
        candidates = candidates.copy()
        yardline_100 = 100 - yardline
        
        candidates['similarity'] = 1.0 / (1.0 + 
            abs(candidates['down'] - down) +
            abs(candidates['ydstogo'] - ydstogo).clip(0, 20) +
            abs(candidates['yardline_100'] - yardline_100).clip(0, 50) * 0.5
        )
        
        # Take top K similar plays
        k = min(200, len(candidates))
        top_k = candidates.nlargest(k, 'similarity')
        
        return top_k
    
    def _sample_run(self, candidates, yardline):
        """Sample run play outcome from skew-t distribution."""
        yards_data = candidates['yards_gained'].values
        yards_data = yards_data[~np.isnan(yards_data)]
        
        if len(yards_data) < 10:
            return self._simple_outcome('rush', 6)
        
        # Fit skew-t parameters using method of moments approximation
        mean = np.mean(yards_data)
        std = np.std(yards_data)
        skew = stats.skew(yards_data) if len(yards_data) > 10 else 0
        
        # Use Biro & Walker defaults if data is insufficient
        if std < 0.1:
            std = SKEWT_DEFAULT['scale']
        
        # Calculate TD probability
        distance_to_goal = 100 - yardline
        td_plays = np.sum(yards_data >= distance_to_goal)
        td_prob = min(0.15, td_plays / len(yards_data) if len(yards_data) > 0 else 0.01)
        
        # Generate outcome
        if self.np_random.random() < td_prob:
            # Touchdown
            return {'yards_gained': float(distance_to_goal), 'turnover': False}
        
        # Sample from t-distribution (approximation of skew-t)
        df = max(2.1, SKEWT_DEFAULT['df'])
        yards = stats.t.rvs(df, loc=mean, scale=std, random_state=self.np_random)
        
        # Clamp to reasonable range
        yards = float(np.clip(yards, -10, distance_to_goal - 1))
        
        # Fumble probability based on historical rate
        fumble_rate = candidates['fumble'].mean() if 'fumble' in candidates.columns else 0.01
        turnover = self.np_random.random() < fumble_rate
        
        return {'yards_gained': yards, 'turnover': bool(turnover)}
    
    def _sample_pass(self, candidates, yardline):
        """Sample pass play outcome from Gamma/categorical mixture."""
        distance_to_goal = 100 - yardline
        
        # Calculate outcome probabilities from data
        n = len(candidates)
        if n == 0:
            return self._simple_outcome('pass', 6)
        
        # Count categorical outcomes
        incompletions = candidates[candidates['incomplete_pass'] == 1] if 'incomplete_pass' in candidates.columns else []
        sacks = candidates[candidates['sack'] == 1] if 'sack' in candidates.columns else []
        interceptions = candidates[candidates['interception'] == 1] if 'interception' in candidates.columns else []
        touchdowns = candidates['yards_gained'] >= distance_to_goal
        
        n_inc = len(incompletions)
        n_sack = len(sacks)
        n_int = len(interceptions)
        n_td = touchdowns.sum()
        n_comp = n - n_inc - n_sack - n_int - n_td
        
        # Dirichlet-Multinomial with prior weighting (Biro & Walker)
        prior_weight = 10
        probs = np.array([n_comp, n_inc, n_sack, n_int, n_td]) + prior_weight
        probs = probs / probs.sum()
        
        # Sample outcome type
        outcome = self.np_random.choice(['completion', 'incomplete', 'sack', 'interception', 'td'], p=probs)
        
        if outcome == 'incomplete':
            return {'yards_gained': 0.0, 'turnover': False}
        
        if outcome == 'sack':
            sack_yards = self.np_random.choice([-3, -5, -7, -10], p=[0.3, 0.35, 0.25, 0.1])
            return {'yards_gained': float(sack_yards), 'turnover': False}
        
        if outcome == 'interception':
            return {'yards_gained': 0.0, 'turnover': True}
        
        if outcome == 'td':
            return {'yards_gained': float(distance_to_goal), 'turnover': False}
        
        # Completion - sample from Gamma
        completions = candidates[(candidates['pass'] == 1) & 
                                  (candidates.get('incomplete_pass', 0) != 1) &
                                  (candidates.get('sack', 0) != 1)]
        
        if len(completions) > 5:
            comp_yards = completions['yards_gained'].values
            comp_yards = comp_yards[comp_yards > 0]  # Positive gains only
            
            if len(comp_yards) > 5:
                # Gamma-Gamma conjugate posterior
                alpha = GAMMA_ALPHA
                beta_prior = GAMMA_PRIOR_BETA / GAMMA_PRIOR_ALPHA
                
                # Update with observed data
                sum_yards = comp_yards.sum()
                n_obs = len(comp_yards)
                beta_post = (GAMMA_PRIOR_BETA + sum_yards) / (GAMMA_PRIOR_ALPHA + n_obs)
                
                yards = stats.gamma.rvs(alpha, scale=beta_post, random_state=self.np_random)
                yards = float(np.clip(yards, 1, distance_to_goal - 1))
                return {'yards_gained': yards, 'turnover': False}
        
        # Fallback: sample from data
        yards = self.np_random.normal(7, 5)
        yards = float(np.clip(yards, 1, distance_to_goal - 1))
        return {'yards_gained': yards, 'turnover': False}
    
    def _simple_outcome(self, play_type, box_count):
        """Simple fallback model when data is insufficient."""
        if play_type == 'pass':
            base_yards = 7.0
            variance = 8.0
            int_prob = 0.02
        else:
            base_yards = 4.0
            variance = 3.0
            int_prob = 0.01
        
        # Box count effects
        if play_type == 'rush':
            base_yards -= (box_count - 6) * 0.5
        else:
            base_yards += (box_count - 6) * 0.3
        
        yards = self.np_random.normal(base_yards, variance)
        yards = float(np.clip(yards, -10, 50))
        turnover = self.np_random.random() < int_prob
        
        return {'yards_gained': yards, 'turnover': bool(turnover)}
