"""
Unified Evaluation Utilities for NFL Agents

Provides standardized evaluation metrics across all agent types.
"""

import numpy as np
import rlcard
from rlcard.agents.random_agent import RandomAgent
from rlcard.utils import tournament


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
    }
    
    if verbose:
        print(f"Off={offense_epa:.3f} | Def={defense_epa:.3f} | Self={self_play_epa:.3f}")
    
    return results


def format_eval_line(episode, results):
    """Format evaluation results as a standard log line."""
    return (f"Episode {episode}: "
            f"Off={results['offense_epa']:.3f} | "
            f"Def={results['defense_epa']:.3f} | "
            f"Self={results['self_play_epa']:.3f}")


def quick_eval(agent, game='nfl-bucketed', num_games=50):
    """Quick evaluation for during training (fewer games)."""
    return evaluate_agent(agent, game, num_games, verbose=False)


class EvalLogger:
    """Logger for tracking evaluation metrics over training."""
    
    def __init__(self, log_path=None):
        self.history = []
        self.log_path = log_path
        
        if log_path:
            with open(log_path, 'w') as f:
                f.write("episode,offense_epa,defense_epa,self_play_epa\n")
    
    def log(self, episode, results):
        """Log evaluation results."""
        self.history.append({
            'episode': episode,
            **results
        })
        
        if self.log_path:
            with open(self.log_path, 'a') as f:
                f.write(f"{episode},{results['offense_epa']:.4f},"
                       f"{results['defense_epa']:.4f},{results['self_play_epa']:.4f}\n")
    
    def get_best(self, metric='offense_epa'):
        """Get best episode by metric."""
        if not self.history:
            return None
        return max(self.history, key=lambda x: x[metric])
