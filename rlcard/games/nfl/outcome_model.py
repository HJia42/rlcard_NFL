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


def skewt_rvs(location, scale, skewness, df, size=1, random_state=None):
    """Sample from skew-t distribution.
    
    Based on Biro & Walker representation using normal/chi-squared mixture.
    
    Args:
        location: Location parameter (mu)
        scale: Scale parameter (sigma)
        skewness: Skewness parameter (alpha)
        df: Degrees of freedom (nu)
        size: Number of samples
        random_state: Numpy random state
        
    Returns:
        Array of samples from skew-t distribution
    """
    rng = random_state if random_state is not None else np.random.default_rng()
    
    # Generate standard normal and chi-squared
    z = rng.standard_normal(size)
    v = rng.chisquare(df, size)
    
    # Skew-normal transformation
    delta = skewness / np.sqrt(1 + skewness**2)
    u = rng.standard_normal(size)
    z_skew = delta * np.abs(u) + np.sqrt(1 - delta**2) * z
    
    # Scale by chi-squared for t-distribution tails
    samples = location + scale * z_skew * np.sqrt(df / v)
    
    return samples


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
        
        # Scale yards_gained by ydstogo ratio (KNN adjustment)
        # If historical play was on 3rd & 15 but current is 3rd & 5,
        # scale yards proportionally
        if 'ydstogo' in top_k.columns and ydstogo > 0:
            top_k = top_k.copy()
            scale_ratio = ydstogo / top_k['ydstogo'].clip(1, None)
            # Apply mild scaling (sqrt to dampen extreme ratios)
            scale_ratio = np.sqrt(scale_ratio).clip(0.5, 2.0)
            top_k['yards_gained'] = top_k['yards_gained'] * scale_ratio
        
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
            # Fallback to skew-t distribution (Biro & Walker)
            mean = np.mean(yards_data[yards_data > 0]) if len(yards_data[yards_data > 0]) > 0 else 4.0
            std = np.std(yards_data[yards_data > 0]) if len(yards_data[yards_data > 0]) > 1 else 3.0
            yards = skewt_rvs(
                location=mean, 
                scale=std/2,
                skewness=SKEWT_DEFAULT['skewness'],
                df=SKEWT_DEFAULT['df'],
                size=1,
                random_state=self.np_random
            )[0]
            yards = float(np.clip(yards, 1, distance_to_goal - 1))
        
        turnover = self.np_random.random() < fumble_rate
        return {'yards_gained': yards, 'turnover': bool(turnover)}
    
    def _fit_sack_distribution(self, candidates):
        """Fit sack yardage distribution from data."""
        if 'sack' not in candidates.columns:
            return np.array([-3, -5, -7, -10]), np.array([0.3, 0.35, 0.25, 0.1])
        
        sacks = candidates[candidates['sack'] == 1]['yards_gained'].values
        sacks = sacks[sacks < 0]  # Only negative yards
        
        if len(sacks) < 10:
            return np.array([-3, -5, -7, -10]), np.array([0.3, 0.35, 0.25, 0.1])
        
        # Bin into categories and compute Dirichlet posterior
        bins = [-15, -10, -7, -5, -3, 0]
        counts = np.histogram(sacks, bins=bins)[0]
        
        # Dirichlet prior (alpha=1 for each bin)
        prior = np.ones(len(counts))
        probs = (counts + prior) / (counts.sum() + prior.sum())
        
        # Representative values for each bin
        bin_centers = np.array([-12, -8, -6, -4, -2])
        
        return bin_centers, probs
    
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
            # Fit sack distribution from data
            sack_yards, sack_probs = self._fit_sack_distribution(candidates)
            sack_yardage = self.np_random.choice(sack_yards, p=sack_probs)
            return {'yards_gained': float(sack_yardage), 'turnover': False}
        
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
