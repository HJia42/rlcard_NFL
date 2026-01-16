"""
Setup script for Cython NFL game extensions.

Usage:
    python setup.py build_ext --inplace

Requirements:
    pip install cython numpy
"""

import os
import sys
from pathlib import Path

try:
    from setuptools import setup, Extension
    from Cython.Build import cythonize
    import numpy as np
    CYTHON_AVAILABLE = True
except ImportError as e:
    CYTHON_AVAILABLE = False
    print(f"Cython build unavailable: {e}")
    print("Install with: pip install cython numpy")


def get_extensions():
    """Get list of Cython extensions to build."""
    cython_dir = Path(__file__).parent / "rlcard" / "games" / "nfl" / "cython"
    
    extensions = []
    
    # game_fast.pyx - Standard NFL game
    game_fast = Extension(
        "rlcard.games.nfl.cython.game_fast",
        sources=[str(cython_dir / "game_fast.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
    )
    extensions.append(game_fast)
    
    # game_iig_fast.pyx - IIG (Imperfect Information Game) variant
    game_iig_fast_path = cython_dir / "game_iig_fast.pyx"
    if game_iig_fast_path.exists():
        game_iig_fast = Extension(
            "rlcard.games.nfl.cython.game_iig_fast",
            sources=[str(game_iig_fast_path)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=["-O3"] if sys.platform != "win32" else ["/O2"],
        )
        extensions.append(game_iig_fast)
    
    return extensions


if CYTHON_AVAILABLE:
    extensions = cythonize(
        get_extensions(),
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    )
else:
    extensions = []


if __name__ == "__main__":
    if not CYTHON_AVAILABLE:
        print("ERROR: Cython not installed. Run: pip install cython numpy")
        sys.exit(1)
    
    setup(
        name="rlcard-nfl-cython",
        version="0.1.0",
        description="Fast Cython NFL game for RLCard",
        ext_modules=extensions,
        zip_safe=False,
    )
