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
            if kick_distance <= 30:
                prob = 0.95
            elif kick_distance >= 60:
                prob = 0.15
            else:
                prob = 0.95 - (kick_distance - 30) * 0.027
            return float(np.clip(prob, 0.0, 1.0))
        
        # B-spline evaluation - uses LOS distance (distance_to_goal)
        prob = interpolate.splev(distance_to_goal, self.fg_model)
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
            # Simple model: net ~40 yards, min 20 (touchback)
            net_yards = 40
            landing = yardline + net_yards
            
            # Touchback if into end zone
            if landing >= 100:
                return 20.0  # Opponent starts at own 20
            
            # Opponent's yardline = where ball lands
            opp_yardline = 100 - landing
            return float(np.clip(opp_yardline, 1.0, 99.0))
        
        # B-spline evaluation
        opp_start_y100 = interpolate.splev(yardline_100, self.punt_model)
        opp_yardline = 100 - opp_start_y100
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
