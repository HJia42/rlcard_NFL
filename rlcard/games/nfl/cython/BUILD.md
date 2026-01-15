# Building Cython NFL Game Extensions

## Requirements

### Windows
- Python 3.8+
- Cython: `pip install cython numpy`
- Microsoft C++ Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

### Linux (Cloud VM / HPC)
```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# Install Python deps
pip install cython numpy
```

## Building

```bash
cd rlcard_NFL
python setup_cython.py build_ext --inplace
```

## Verification

```python
from rlcard.games.nfl.cython import CYTHON_AVAILABLE, make_fast_game

print(f"Cython available: {CYTHON_AVAILABLE}")
game = make_fast_game(single_play=True)
print(f"Game type: {type(game).__name__}")
# Should print: NFLGameFast (compiled) or NFLGame (fallback)
```

## Usage in Training

The Cython game is automatically used when available:
```python
from rlcard.games.nfl.cython import make_fast_game

# Creates NFLGameFast if compiled, else NFLGame
game = make_fast_game(single_play=True, use_cached_model=True)
```

## Expected Speedup

| Agent Type | Python | Cython | Speedup |
|------------|--------|--------|---------|
| MCCFR | ~2000 iter/s | ~10000 iter/s | 5x |
| CFR | ~1500 iter/s | ~8000 iter/s | 5x |
| PPO* | ~150 ep/s | ~160 ep/s | ~1.1x |

*Neural agents bottlenecked by network, not game simulation
