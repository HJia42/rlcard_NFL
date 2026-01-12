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
MIN_PLAYS_THRESHOLD = 50


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
            if len(form_matches) > 50:
                candidates = form_matches
        
        # Filter by box count
        if 'defenders_in_box' in candidates.columns:
            box_matches = candidates[candidates['defenders_in_box'] == box_count]
            if len(box_matches) > 50:
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
        """Sample run play outcome using categorical + empirical distribution."""
        yards_data = candidates['yards_gained'].values
        yards_data = yards_data[~np.isnan(yards_data)]
        
        if len(yards_data) < 10:
            return self._simple_outcome('rush', 6)
        
        distance_to_goal = 100 - yardline
        
        # Calculate outcome category probabilities from historical data
        n_total = len(yards_data)
        n_loss = int((yards_data < 0).sum())
        n_zero = int((yards_data == 0).sum())
        n_td = int((yards_data >= distance_to_goal).sum())
        n_positive = n_total - n_loss - n_zero - n_td
        
        # Dirichlet-Multinomial with small prior
        prior_weight = 5
        counts = np.array([n_positive, n_zero, n_loss, n_td], dtype=float)
        probs = (counts + prior_weight) / (counts.sum() + 4 * prior_weight)
        
        # Sample outcome category
        outcome = self.np_random.choice(['positive', 'zero', 'loss', 'td'], p=probs)
        
        # Fumble probability
        fumble_rate = candidates['fumble'].mean() if 'fumble' in candidates.columns else 0.01
        
        if outcome == 'td':
            return {'yards_gained': float(distance_to_goal), 'turnover': False}
        
        if outcome == 'zero':
            turnover = self.np_random.random() < fumble_rate
            return {'yards_gained': 0.0, 'turnover': bool(turnover)}
        
        if outcome == 'loss':
            # Sample from historical losses
            losses = yards_data[yards_data < 0]
            if len(losses) > 0:
                yards = float(self.np_random.choice(losses))
            else:
                yards = float(self.np_random.choice([-1, -2, -3, -4]))
            turnover = self.np_random.random() < fumble_rate
            return {'yards_gained': yards, 'turnover': bool(turnover)}
        
        # Positive gain - sample from historical positive gains
        positive_gains = yards_data[(yards_data > 0) & (yards_data < distance_to_goal)]
        if len(positive_gains) > 0:
            yards = float(self.np_random.choice(positive_gains))
        else:
            # Fallback to t-distribution
            mean = np.mean(yards_data[yards_data > 0])
            std = np.std(yards_data[yards_data > 0])
            yards = max(1, stats.t.rvs(5, loc=mean, scale=std/2, random_state=self.np_random))
            yards = float(np.clip(yards, 1, distance_to_goal - 1))
        
        turnover = self.np_random.random() < fumble_rate
        return {'yards_gained': yards, 'turnover': bool(turnover)}
    
    def _sample_pass(self, candidates, yardline):
        """Sample pass play outcome from Gamma/categorical mixture."""
        distance_to_goal = 100 - yardline
        
        # Calculate outcome probabilities from data
        n = len(candidates)
        if n == 0:
            return self._simple_outcome('pass', 6)
        
        # Count categorical outcomes using available columns
        yards = candidates['yards_gained'].values
        
        # Incompletions: passes with exactly 0 yards gained (not negative/sack)
        n_inc = int((yards == 0).sum())
        
        # Sacks: passes with negative yards 
        n_sack = int((yards < 0).sum())
        
        # Interceptions
        n_int = int(candidates['interception'].sum()) if 'interception' in candidates.columns else 0
        
        # Touchdowns: plays reaching the end zone
        n_td = int((yards >= distance_to_goal).sum())
        
        # Completions: the rest (positive yards < touchdown)
        n_comp = int(((yards > 0) & (yards < distance_to_goal)).sum())
        
        # Dirichlet-Multinomial with prior weighting (Biro & Walker)
        prior_weight = 10
        counts = np.array([n_comp, n_inc, n_sack, n_int, n_td], dtype=float)
        probs = (counts + prior_weight) / (counts.sum() + 5 * prior_weight)
        
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
