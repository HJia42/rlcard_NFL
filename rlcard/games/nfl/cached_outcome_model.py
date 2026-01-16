"""
Cached Outcome Model for NFL Play Simulation

Pre-computes outcome distributions for all state combinations at startup,
enabling O(1) lookup during training instead of O(n) filtering.

Supports both bucketed and full-game state spaces.
"""

import numpy as np
import pickle
from pathlib import Path
from collections import defaultdict


# State space configurations
BUCKETED_CONFIG = {
    'formations': ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY'],
    'play_types': ['rush', 'pass'],
    'box_counts': [4, 5, 6, 7, 8],  # Matches game action space (defense actions 0-4 map to 4-8 box)
    'yardline_bins': list(range(5, 100, 5)),  # 5, 10, 15, ..., 95
    'down_values': [1, 2, 3, 4],
    'distance_bins': [1, 3, 6, 10, 20],  # Represents: 1, 2-3, 4-6, 7-10, 11+
}

# Full game uses finer granularity for yardline
FULL_GAME_CONFIG = {
    'formations': ['SHOTGUN', 'SINGLEBACK', 'UNDER CENTER', 'I_FORM', 'EMPTY'],
    'play_types': ['rush', 'pass'],
    'box_counts': [4, 5, 6, 7, 8],  # Matches game action space (defense actions 0-4 map to 4-8 box)
    'yardline_bins': list(range(1, 100, 1)),  # Every yard
    'down_values': [1, 2, 3, 4],
    'distance_bins': [1, 2, 3, 5, 7, 10, 15, 20],  # Finer distance bins
}

# Minimum plays required for reliable estimation
MIN_PLAYS_FOR_CACHE = 10


class CachedOutcomeModel:
    """
    Pre-computed outcome distributions for O(1) sampling.
    
    Caches:
    - Category probabilities for each state combination
    - Sample pools for empirical sampling
    - Fallback to global averages for unseen states
    """
    
    def __init__(self, play_data, np_random, use_bucketed=True, cache_path=None):
        """
        Initialize cached outcome model.
        
        Args:
            play_data: DataFrame of historical plays
            np_random: NumPy random state
            use_bucketed: If True, use bucketed state space (fewer states)
            cache_path: Path to cached model file (optional)
        """
        self.play_data = play_data
        self.np_random = np_random
        self.use_bucketed = use_bucketed
        self.config = BUCKETED_CONFIG if use_bucketed else FULL_GAME_CONFIG
        
        # Cache dictionaries
        self.run_cache = {}   # (formation, box, yardline_bin, down, distance_bin) -> stats
        self.pass_cache = {}  # Same key structure
        
        # Global fallbacks
        self.global_run_stats = None
        self.global_pass_stats = None
        
        # Try to load from cache
        if cache_path and Path(cache_path).exists():
            self._load_cache(cache_path)
        else:
            self._build_cache()
            if cache_path:
                self._save_cache(cache_path)
    
    def _bin_yardline(self, yardline):
        """Bin yardline to nearest configured bin."""
        bins = self.config['yardline_bins']
        for b in bins:
            if yardline <= b:
                return b
        return bins[-1]
    
    def _bin_distance(self, distance):
        """Bin distance to nearest configured bin."""
        bins = self.config['distance_bins']
        for b in bins:
            if distance <= b:
                return b
        return bins[-1]
    
    def _build_cache(self):
        """Pre-compute distributions for all state combinations."""
        print("Building outcome distribution cache...")
        
        # Separate run and pass plays
        run_plays = self.play_data[self.play_data['pass'] == 0].copy()
        pass_plays = self.play_data[self.play_data['pass'] == 1].copy()
        
        # Add binned columns
        run_plays['yl_bin'] = run_plays['yardline_100'].apply(self._bin_yardline)
        run_plays['dist_bin'] = run_plays['ydstogo'].apply(self._bin_distance)
        pass_plays['yl_bin'] = pass_plays['yardline_100'].apply(self._bin_yardline)
        pass_plays['dist_bin'] = pass_plays['ydstogo'].apply(self._bin_distance)
        
        # Compute global fallbacks
        self.global_run_stats = self._compute_run_stats(run_plays)
        self.global_pass_stats = self._compute_pass_stats(pass_plays)
        
        # Build cache for each combination
        total_combos = (
            len(self.config['formations']) * 
            len(self.config['box_counts']) * 
            len(self.config['yardline_bins']) * 
            len(self.config['down_values']) * 
            len(self.config['distance_bins'])
        )
        
        filled = 0
        for formation in self.config['formations']:
            for box in self.config['box_counts']:
                for yl_bin in self.config['yardline_bins']:
                    for down in self.config['down_values']:
                        for dist_bin in self.config['distance_bins']:
                            key = (formation, box, yl_bin, down, dist_bin)
                            
                            # Run plays
                            run_subset = run_plays[
                                (run_plays['offense_formation'] == formation) &
                                (run_plays['defenders_in_box'] == box) &
                                (run_plays['yl_bin'] == yl_bin) &
                                (run_plays['down'] == down) &
                                (run_plays['dist_bin'] == dist_bin)
                            ]
                            if len(run_subset) >= MIN_PLAYS_FOR_CACHE:
                                self.run_cache[key] = self._compute_run_stats(run_subset)
                                filled += 1
                            
                            # Pass plays
                            pass_subset = pass_plays[
                                (pass_plays['offense_formation'] == formation) &
                                (pass_plays['defenders_in_box'] == box) &
                                (pass_plays['yl_bin'] == yl_bin) &
                                (pass_plays['down'] == down) &
                                (pass_plays['dist_bin'] == dist_bin)
                            ]
                            if len(pass_subset) >= MIN_PLAYS_FOR_CACHE:
                                self.pass_cache[key] = self._compute_pass_stats(pass_subset)
                                filled += 1
        
        mode = "bucketed" if self.use_bucketed else "full"
        print(f"  Cache built ({mode}): {len(self.run_cache)} run + {len(self.pass_cache)} pass entries")
        print(f"  Coverage: {filled}/{total_combos*2:.0f} combinations ({100*filled/(total_combos*2):.1f}%)")
    
    def _compute_run_stats(self, df):
        """Compute run play statistics from DataFrame."""
        if len(df) == 0:
            return None
        
        yards = df['yards_gained'].values
        
        # Category probabilities
        n = len(yards)
        p_positive = np.sum(yards > 0) / n
        p_zero = np.sum(yards == 0) / n
        p_loss = np.sum(yards < 0) / n
        
        # Fumble rate
        if 'fumble_lost' in df.columns:
            p_fumble = df['fumble_lost'].mean()
        else:
            p_fumble = 0.01  # ~1% fumble rate
        
        # Sample pools for each category
        positive_yards = yards[yards > 0]
        loss_yards = yards[yards < 0]
        
        return {
            'probs': [p_positive, p_zero, p_loss, p_fumble],
            'positive_pool': positive_yards if len(positive_yards) > 0 else np.array([3.0]),
            'loss_pool': loss_yards if len(loss_yards) > 0 else np.array([-2.0]),
            'mean': yards.mean(),
            'std': yards.std() if len(yards) > 1 else 3.0,
            'n_samples': n,
        }
    
    def _compute_pass_stats(self, df):
        """Compute pass play statistics from DataFrame."""
        if len(df) == 0:
            return None
        
        yards = df['yards_gained'].values
        n = len(yards)
        
        # Category probabilities based on yards
        # Completion: yards > 0
        # Incompletion: yards == 0
        # Sack: yards < 0
        p_complete = np.sum(yards > 0) / n
        p_incomplete = np.sum(yards == 0) / n
        p_sack = np.sum(yards < 0) / n
        
        # Interception rate
        if 'interception' in df.columns:
            p_int = df['interception'].mean()
        else:
            p_int = 0.02  # ~2% int rate
        
        # Sample pool for completions
        completion_yards = yards[yards > 0]
        sack_yards = yards[yards < 0]
        
        return {
            'probs': [p_complete, p_incomplete, p_sack, p_int],
            'completion_pool': completion_yards if len(completion_yards) > 0 else np.array([7.0]),
            'sack_pool': sack_yards if len(sack_yards) > 0 else np.array([-5.0]),
            'mean': yards.mean(),
            'std': yards.std() if len(yards) > 1 else 8.0,
            'n_samples': n,
        }
    
    def sample_run(self, formation, box_count, yardline, down, distance, distance_to_goal):
        """
        Sample run play outcome using cached distribution.
        
        Returns:
            dict with 'yards_gained', 'turnover', 'touchdown'
        """
        # Get cached stats
        key = (formation, box_count, self._bin_yardline(yardline), 
               down, self._bin_distance(distance))
        stats = self.run_cache.get(key, self.global_run_stats)
        
        if stats is None:
            stats = self.global_run_stats
        
        # Sample category
        p_positive, p_zero, p_loss, p_fumble = stats['probs']
        
        # Check for fumble first
        if self.np_random.random() < p_fumble:
            yards = int(self.np_random.choice(stats['positive_pool'])) if len(stats['positive_pool']) > 0 else 3
            return {'yards_gained': yards, 'turnover': True, 'touchdown': False}
        
        # Sample outcome category
        r = self.np_random.random()
        if r < p_positive:
            yards = int(self.np_random.choice(stats['positive_pool']))
        elif r < p_positive + p_zero:
            yards = 0
        else:
            yards = int(self.np_random.choice(stats['loss_pool']))
        
        # Check touchdown
        touchdown = yards >= distance_to_goal
        
        return {'yards_gained': yards, 'turnover': False, 'touchdown': touchdown}
    
    def sample_pass(self, formation, box_count, yardline, down, distance, distance_to_goal):
        """
        Sample pass play outcome using cached distribution.
        
        Returns:
            dict with 'yards_gained', 'turnover', 'touchdown'
        """
        # Get cached stats
        key = (formation, box_count, self._bin_yardline(yardline), 
               down, self._bin_distance(distance))
        stats = self.pass_cache.get(key, self.global_pass_stats)
        
        if stats is None:
            stats = self.global_pass_stats
        
        # Sample category
        p_complete, p_incomplete, p_sack, p_int = stats['probs']
        
        # Check for interception first
        if self.np_random.random() < p_int:
            return {'yards_gained': 0, 'turnover': True, 'touchdown': False}
        
        # Sample outcome category
        r = self.np_random.random()
        if r < p_complete:
            yards = int(self.np_random.choice(stats['completion_pool']))
        elif r < p_complete + p_incomplete:
            yards = 0
        else:
            yards = int(self.np_random.choice(stats['sack_pool']))
        
        # Check touchdown
        touchdown = yards >= distance_to_goal
        
        return {'yards_gained': yards, 'turnover': False, 'touchdown': touchdown}
    
    def sample(self, formation, play_type, box_count, yardline, down, distance):
        """
        Sample play outcome (main interface).
        
        Args:
            formation: Offensive formation
            play_type: 'rush' or 'pass'
            box_count: Number of defenders in box
            yardline: Yard line (1-99 from own goal)
            down: Current down (1-4)
            distance: Yards to first down
            
        Returns:
            dict with 'yards_gained', 'turnover', 'touchdown'
        """
        distance_to_goal = 100 - yardline
        
        if play_type == 'rush':
            return self.sample_run(formation, box_count, yardline, down, distance, distance_to_goal)
        else:
            return self.sample_pass(formation, box_count, yardline, down, distance, distance_to_goal)
    
    def _save_cache(self, path):
        """Save cache to disk."""
        data = {
            'run_cache': self.run_cache,
            'pass_cache': self.pass_cache,
            'global_run_stats': self.global_run_stats,
            'global_pass_stats': self.global_pass_stats,
            'config': self.config,
            'use_bucketed': self.use_bucketed,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved cache to {path}")
    
    def _load_cache(self, path):
        """Load cache from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.run_cache = data['run_cache']
        self.pass_cache = data['pass_cache']
        self.global_run_stats = data['global_run_stats']
        self.global_pass_stats = data['global_pass_stats']
        self.config = data['config']
        self.use_bucketed = data['use_bucketed']
        
        mode = "bucketed" if self.use_bucketed else "full"
        print(f"Loaded cached outcome model ({mode}): {len(self.run_cache)} run + {len(self.pass_cache)} pass entries")


# Default cache paths
DEFAULT_BUCKETED_CACHE = Path(__file__).parent / "cached_outcomes_bucketed.pkl"
DEFAULT_FULL_CACHE = Path(__file__).parent / "cached_outcomes_full.pkl"


def get_cached_outcome_model(play_data, np_random, use_bucketed=True):
    """
    Get or create a cached outcome model.
    
    Args:
        play_data: Historical play DataFrame
        np_random: NumPy random state
        use_bucketed: Whether to use bucketed state space
        
    Returns:
        CachedOutcomeModel instance
    """
    cache_path = DEFAULT_BUCKETED_CACHE if use_bucketed else DEFAULT_FULL_CACHE
    return CachedOutcomeModel(play_data, np_random, use_bucketed, cache_path)
