"""
Unified Evaluation Utilities for NFL Agents

Provides standardized evaluation metrics across all agent types.
"""

import csv
from typing import Dict, Iterable, Optional

import numpy as np
import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament

EVAL_COLUMNS = (
    'episode',
    'offense_epa',
    'defense_epa',
    'self_play_epa',
)


def evaluate_agent(agent, game='nfl-bucketed', num_games=200, verbose=True):
    """
    Evaluate an agent with standardized metrics.
    
    Args:
        agent: Agent to evaluate (must have eval_step method)
        game: Game environment ('nfl' or 'nfl-bucketed')
        num_games: Number of games for each evaluation
        verbose: Print results
        
    Returns:
        dict with keys:
            - offense_epa: Agent's EPA when playing offense vs random
            - defense_epa: Agent's EPA when playing defense vs random
            - self_play_epa: Agent's EPA when playing against itself
    """
    env = rlcard.make(game, config={'single_play': True})
    random_agent = RandomAgent(num_actions=env.num_actions)
    
    # Offense vs Random (agent is player 0)
    env.set_agents([agent, random_agent])
    off_result = tournament(env, num_games)
    offense_epa = off_result[0]
    
    # Defense vs Random (agent is player 1)
    env.set_agents([random_agent, agent])
    def_result = tournament(env, num_games)
    defense_epa = def_result[1]
    
    # Self-play (agent vs itself)
    env.set_agents([agent, agent])
    self_result = tournament(env, num_games)
    self_play_epa = self_result[0]
    
    results = {
        'offense_epa': offense_epa,
        'defense_epa': defense_epa,
        'self_play_epa': self_play_epa,
        'num_games': num_games,
        'game': game,
    }
    
    if verbose:
        print(f"Off={offense_epa:.3f} | Def={defense_epa:.3f} | Self={self_play_epa:.3f} | N={num_games}")
    
    return results


def format_eval_line(episode, results):
    """Format evaluation results as a standard log line."""
    base = (f"Episode {episode}: "
            f"Off={results['offense_epa']:.3f} | "
            f"Def={results['defense_epa']:.3f} | "
            f"Self={results['self_play_epa']:.3f}")
    if 'num_games' in results:
        return f"{base} | N={results['num_games']}"
    return base


def format_metrics(metrics: Dict[str, object], precision: int = 3) -> str:
    """Format metrics dict into a standardized key=value string."""
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}={value:.{precision}f}")
        else:
            formatted.append(f"{key}={value}")
    return " | ".join(formatted)


def format_step_line(label: str, step: object, metrics: Dict[str, object], precision: int = 3) -> str:
    """Format a standardized training line for logging progress."""
    return f"{label} {step}: {format_metrics(metrics, precision=precision)}"


def quick_eval(agent, game='nfl-bucketed', num_games=50):
    """Quick evaluation for during training (fewer games)."""
    return evaluate_agent(agent, game, num_games, verbose=False)


class EvalLogger:
    """Logger for tracking evaluation metrics over training."""
    
    def __init__(self, log_path=None):
        self.history = []
        self.log_path = log_path
        
        if log_path:
            with open(log_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(EVAL_COLUMNS)
    
    def log(self, episode, results):
        """Log evaluation results."""
        record = {
            'episode': episode,
            'offense_epa': results['offense_epa'],
            'defense_epa': results['defense_epa'],
            'self_play_epa': results['self_play_epa'],
        }
        self.history.append(record)
        
        if self.log_path:
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    episode,
                    f"{results['offense_epa']:.4f}",
                    f"{results['defense_epa']:.4f}",
                    f"{results['self_play_epa']:.4f}",
                ])
    
    def get_best(self, metric='offense_epa'):
        """Get best episode by metric."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x[metric])


def load_eval_history(csv_path: str) -> Dict[str, Iterable[float]]:
    """Load evaluation history from a CSV log created by EvalLogger."""
    history = {key: [] for key in EVAL_COLUMNS}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history['episode'].append(int(row['episode']))
            history['offense_epa'].append(float(row['offense_epa']))
            history['defense_epa'].append(float(row['defense_epa']))
            history['self_play_epa'].append(float(row['self_play_epa']))
    return history


def plot_eval_history(csv_path: str, save_path: Optional[str] = None, title: Optional[str] = None):
    """Plot standardized evaluation history from an EvalLogger CSV."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not available, skipping eval plot")
        return None

    history = load_eval_history(csv_path)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['episode'], history['offense_epa'], label='Offense EPA')
    ax.plot(history['episode'], history['defense_epa'], label='Defense EPA')
    ax.plot(history['episode'], history['self_play_epa'], label='Self-play EPA')
    ax.set_xlabel('Episode')
    ax.set_ylabel('EPA')
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved evaluation plot to {save_path}")
    return fig
