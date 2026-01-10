"""
Special Teams Engine for NFL Game

Uses B-spline interpolation models for field goal probability and punt outcomes.
Models are fitted from historical NFL data and saved for reuse.
"""

import numpy as np
import pickle
from pathlib import Path

try:
    import scipy.interpolate as interpolate
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not installed, using simplified special teams model")


# Default model path
DEFAULT_MODEL_PATH = Path(__file__).parent / "special_teams_models.pkl"


class SpecialTeamsEngine:
    """Engine for special teams play outcomes (FG and Punt)."""
    
    def __init__(self, model_path=None):
        """Initialize special teams engine.
        
        Args:
            model_path: Path to saved B-spline models (optional)
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.punt_model = None  # (t, c, k) tuple for BSpline
        self.fg_model = None    # (t, c, k) tuple for BSpline
        self.use_simple_model = not HAS_SCIPY
        
    def load_models(self):
        """Load saved B-spline models from disk."""
        if not HAS_SCIPY:
            print("Using simplified special teams model (scipy not available)")
            self.use_simple_model = True
            return
            
        if self.model_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.punt_model = data.get('punt')
                    self.fg_model = data.get('fg')
                print(f"Loaded Special Teams models from {self.model_path}")
            except Exception as e:
                print(f"Error loading models: {e}, using simplified model")
                self.use_simple_model = True
        else:
            print(f"No saved models at {self.model_path}, using simplified model")
            self.use_simple_model = True

    def save_models(self):
        """Save B-spline models to disk."""
        if self.punt_model is None or self.fg_model is None:
            print("No models to save")
            return
            
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'punt': self.punt_model,
                'fg': self.fg_model
            }, f)
        print(f"Saved models to {self.model_path}")

    def fetch_data(self, years=range(2010, 2025)):
        """Fetch special teams data from nfl_data_py.
        
        Args:
            years: Range of seasons to fetch (default 2010-2024)
        
        Returns:
            DataFrame with punt and FG plays
        """
        try:
            import nfl_data_py as nfl
        except ImportError:
            print("Error: nfl_data_py not installed. Run: pip install nfl_data_py")
            return None
        
        print(f"Fetching Special Teams data for {list(years)}...")
        cols = ['season', 'play_type', 'yardline_100', 'kick_distance', 
                'field_goal_result', 'return_yards', 'touchback', 'punt_blocked']
        
        try:
            df = nfl.import_pbp_data(years, columns=cols, downcast=True)
            print(f"Data Loaded: {len(df):,} rows.")
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def fit_models(self, df, s_fg=None, s_punt=None):
        """Fit B-spline models for FG and Punt.
        
        Args:
            df: DataFrame from fetch_data
            s_fg: Smoothing parameter for FG spline (default: len(x))
            s_punt: Smoothing parameter for punt spline (default: len(x)*5)
        """
        if not HAS_SCIPY:
            print("Cannot fit models without scipy")
            return
            
        if df is None:
            print("No data provided")
            return

        # --- Field Goals ---
        print("Fitting Field Goal Model (B-Spline) on kick distance...")
        fg_data = df[(df['play_type'] == 'field_goal') & (df['kick_distance'].notnull())].copy()
        
        if len(fg_data) == 0:
            print("No FG data found")
            return
        
        fg_data['success'] = np.where(fg_data['field_goal_result'] == 'made', 1, 0)
        
        # Aggregate by kick distance
        fg_stats = fg_data.groupby('kick_distance')['success'].agg(['mean', 'count'])
        fg_stats = fg_stats[fg_stats['count'] >= 10]  # Filter for sample size
        
        x_fg = list(fg_stats.index.values)
        y_fg = list(fg_stats['mean'].values)
        w_fg = list(np.sqrt(fg_stats['count'].values))
        
        # Force probability to 0 at long distances (>= 65 yards)
        synthetic_dists = [65, 68, 70, 75, 80]
        for d in synthetic_dists:
            x_fg.append(d)
            y_fg.append(0.0)
            w_fg.append(50.0)  # High weight to enforce
            
        # Re-sort
        sorted_indices = np.argsort(x_fg)
        x_fg = np.array(x_fg)[sorted_indices]
        y_fg = np.array(y_fg)[sorted_indices]
        w_fg = np.array(w_fg)[sorted_indices]
        
        s_fg_val = s_fg if s_fg is not None else len(x_fg)
        self.fg_model = interpolate.splrep(x_fg, y_fg, w=w_fg, s=s_fg_val)
        print(f"  FG model fitted on {len(fg_stats)} unique distances")
        
        # --- Punts ---
        print("Fitting Punt Model (B-Spline)...")
        punt_data = df[(df['play_type'] == 'punt') & (df['yardline_100'].notnull())].copy()
        
        if len(punt_data) == 0:
            print("No punt data found")
            return
        
        punt_data['return_yards'] = punt_data['return_yards'].fillna(0)
        punt_data['kick_distance'] = punt_data['kick_distance'].fillna(0)
        
        def calc_opp_start(row):
            if row.get('touchback', 0) == 1:
                return 80  # Touchback -> Opp starts at own 20 (80 from goal)
            
            landing_spot = row['yardline_100'] - row['kick_distance']
            final_spot = landing_spot + row['return_yards']
            return 100 - final_spot
            
        punt_data['opp_start'] = punt_data.apply(calc_opp_start, axis=1)
        punt_data = punt_data[(punt_data['opp_start'] > 0) & (punt_data['opp_start'] < 100)]
        
        # Aggregate by yardline
        punt_stats = punt_data.groupby('yardline_100')['opp_start'].agg(['mean', 'count'])
        punt_stats = punt_stats[punt_stats['count'] >= 10]
        
        x_pt = list(punt_stats.index.values)
        y_pt = list(punt_stats['mean'].values)
        w_pt = list(np.sqrt(punt_stats['count'].values))
        
        # Add synthetic boundary points to smooth edge behavior
        # yardline_100 = yards to opponent's goal (high = deep in own territory)
        # When punting from very close to opponent's goal (yardline_100 < 30),
        # opponent should get ball fairly deep in their territory
        synthetic_pts = [(10, 75), (5, 80)]  # yardline_100 -> opp_start (yards to their goal)
        for x_val, y_val in synthetic_pts:
            x_pt.append(x_val)
            y_pt.append(y_val)
            w_pt.append(30.0)  # Moderate weight
        
        # Re-sort by x
        sorted_indices = np.argsort(x_pt)
        x_pt = np.array(x_pt)[sorted_indices]
        y_pt = np.array(y_pt)[sorted_indices]
        w_pt = np.array(w_pt)[sorted_indices]
        
        s_punt_val = s_punt if s_punt is not None else len(x_pt) * 5
        self.punt_model = interpolate.splrep(x_pt, y_pt, w=w_pt, s=s_punt_val)
        print(f"  Punt model fitted on {len(punt_stats)} unique yardlines")
        
        self.use_simple_model = False
        self.save_models()
        self.validate_models()

    def fetch_and_fit(self, years=range(2010, 2025), s_fg=None, s_punt=None):
        """Convenience method to fetch data and fit models."""
        df = self.fetch_data(years)
        if df is not None:
            self.fit_models(df, s_fg=s_fg, s_punt=s_punt)

    def validate_models(self):
        """Print sample predictions to validate models."""
        print("\n--- Model Validation ---")
        
        # FG Check (using kick distance now)
        print("FG Probabilities:")
        for yl in [90, 80, 70, 60, 50]:  # yardline from own goal
            prob = self.predict_fg_prob(yl)
            kick_dist = (100 - yl) + 17
            print(f"  At opp {100-yl} (kick {kick_dist}yd): {prob:.1%}")
            
        # Punt Check
        print("Punt Outcomes (Opponent Start):")
        for yl in [25, 35, 45, 55]:  # yardline from own goal
            opp = self.predict_punt_outcome(yl)
            print(f"  Punt from own {yl}: Opp starts at own {opp:.0f}")

    def predict_fg_prob(self, yardline):
        """Predict field goal success probability.
        
        Args:
            yardline: Yards from OWN goal line (1-99)
                      e.g., 75 means at opponent's 25, a makeable FG
                      e.g., 25 means at own 25, way too far for FG
            
        Returns:
            Probability of successful field goal (0-1)
        """
        # Distance to opponent's goal = 100 - yardline
        # Kick distance = distance to goal + 17 (end zone + snap)
        distance_to_goal = 100 - yardline
        kick_distance = distance_to_goal + 17
        
        if self.use_simple_model or self.fg_model is None:
            # Simple linear model: 
            # At opp 20 (yardline=80, kick=37yd): ~90%
            # At opp 50 (yardline=50, kick=67yd): ~20%
            return float(np.clip(self._simple_fg_prob(kick_distance), 0.0, 1.0))
        
        # B-spline evaluation - uses kick distance (LOS + 17 yards)
        prob = interpolate.splev(kick_distance, self.fg_model)
        if not np.isfinite(prob):
            return float(np.clip(self._simple_fg_prob(kick_distance), 0.0, 1.0))
        return float(np.clip(prob, 0.0, 1.0))
        
    def predict_punt_outcome(self, yardline):
        """Predict punt outcome - where opponent gets the ball.
        
        Args:
            yardline: Yards from opponent's goal line (1-99)
            
        Returns:
            Opponent's starting yardline (yards from their goal)
        """
        yardline_100 = 100 - yardline  # Convert to yards from own goal
        
        if self.use_simple_model or self.punt_model is None:
            opp_yardline = self._simple_punt_outcome(yardline)
            return float(np.clip(opp_yardline, 1.0, 99.0))
        
        # B-spline evaluation
        opp_start_y100 = interpolate.splev(yardline_100, self.punt_model)
        if not np.isfinite(opp_start_y100):
            return float(np.clip(self._simple_punt_outcome(yardline), 1.0, 99.0))
        opp_yardline = 100 - opp_start_y100
        return float(np.clip(opp_yardline, 1.0, 99.0))

    @staticmethod
    def _simple_fg_prob(kick_distance):
        """Simple linear FG model based on kick distance."""
        if kick_distance <= 30:
            prob = 0.95
        elif kick_distance >= 60:
            prob = 0.15
        else:
            prob = 0.95 - (kick_distance - 30) * 0.027
        return float(np.clip(prob, 0.0, 1.0))

    @staticmethod
    def _simple_punt_outcome(yardline):
        """Simple punt model: net ~40 yards, min 20 (touchback)."""
        net_yards = 40
        landing = yardline + net_yards

        # Touchback if into end zone
        if landing >= 100:
            return 20.0

        # Opponent's yardline = where ball lands
        opp_yardline = 100 - landing
        return float(np.clip(opp_yardline, 1.0, 99.0))
    
    def calculate_fg_epa(self, yardline, current_ep):
        """Calculate EPA for a field goal attempt.
        
        Args:
            yardline: Yards from opponent's goal line
            current_ep: Current expected points before the play
            
        Returns:
            Expected EPA from attempting field goal
        """
        success_prob = self.predict_fg_prob(yardline)
        
        # If successful: +3 points
        # If missed: opponent gets ball at yardline (or 20)
        miss_yardline = max(yardline, 20)  # Opponent gets ball at LOS or 20
        miss_opp_ep = (miss_yardline / 100) * 3  # Rough opponent EP
        
        # EPA = P(make) * 3 - P(miss) * opponent_ep - current_ep
        expected_outcome = success_prob * 3 - (1 - success_prob) * miss_opp_ep
        return expected_outcome - current_ep
    
    def calculate_punt_epa(self, yardline, current_ep):
        """Calculate EPA for a punt.
        
        Args:
            yardline: Yards from opponent's goal line
            current_ep: Current expected points before the play
            
        Returns:
            Expected EPA from punting
        """
        opp_yardline = self.predict_punt_outcome(yardline)
        
        # Opponent's EP from their starting position
        opp_ep = (opp_yardline / 100) * 3  # Rough EP estimate
        
        # Our EPA = -opponent_ep - current_ep (flip perspective)
        return -opp_ep - current_ep


# Global instance for shared use
_special_teams_engine = None

def get_special_teams_engine():
    """Get or create the global special teams engine."""
    global _special_teams_engine
    if _special_teams_engine is None:
        _special_teams_engine = SpecialTeamsEngine()
        _special_teams_engine.load_models()
    return _special_teams_engine
