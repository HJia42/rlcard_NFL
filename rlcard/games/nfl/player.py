"""
NFL Player Classes for RLCard

Player 0: Offense
Player 1: Defense
"""

class OffensePlayer:
    """Offensive player who selects formation and play type."""
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.status = 'alive'
    
    def get_state(self, down, ydstogo, yardline, legal_actions):
        """Get offense state (doesn't see defense alignment)."""
        return {
            'down': down,
            'ydstogo': ydstogo,
            'yardline': yardline,
            'legal_actions': legal_actions,
            'player_id': self.player_id
        }


class DefensePlayer:
    """Defensive player who sees formation and selects alignment."""
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.status = 'alive'
    
    def get_state(self, down, ydstogo, yardline, formation, legal_actions):
        """Get defense state (sees offense formation)."""
        return {
            'down': down,
            'ydstogo': ydstogo,
            'yardline': yardline,
            'formation': formation,
            'legal_actions': legal_actions,
            'player_id': self.player_id
        }
