# Cython NFL Game Module
"""
Fast Cython implementations of NFL game logic.

To compile:
    python setup.py build_ext --inplace

If Cython is not installed or compilation fails, 
the pure Python versions are used as fallback.
"""

try:
    from .game_fast import NFLGameFast, make_fast_game
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    
    def make_fast_game(single_play=True, use_cached_model=True, seed=None):
        """Fallback to Python game."""
        from rlcard.games.nfl.game import NFLGame
        return NFLGame(
            single_play=single_play, 
            use_cached_model=use_cached_model
        )
